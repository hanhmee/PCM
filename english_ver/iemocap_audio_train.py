import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F

from iemocap_audio_model import AudioClassifier
#from iemocap_audio_dataset import data_loader
from audio_parsing_dataset import EMOTION_LABELS, EMOTION_LABELS_VI, data_loader
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import pandas as pd
import random

import os 
import torch.nn as nn
import numpy as np

import logging

import argparse
from argparse import ArgumentParser
import json
import gc

from scipy.signal import resample

class Trainer:
    def __init__(self,args):
        
        self._check_args(args)
        self.args = args
        self.classification = getattr(args, "objective", "regression") == "classification"
        self.set_seed(self.args.seed)
        
        self.logger = None
        self.vad_model = AudioClassifier(self.args).cuda()

        self.inferenced_value = []
        self.epoch_metrics = []
        
        self.best_score = float("-inf") if self.classification else 0
        self.save_path = self.args.load_model_path
        self.logger_path = self.args.logger_path
    
    def _check_args(self, args):
        assert args.task in ['normal','temporal_normal','temporal_intonation_and_wav2vec2','interpolated_intonation_and_wav2vec2','msp_normal','msp_intonation_and_wav2vec2','contour_image_vad','audio_parsing','audio_parsing_probing','ctc_normal','contour_image_and_ctc','intonation_contour','normal_with_text','audio_parsing_with_text','audio_parsing_with_text_concat','intonation_contour_with_text','intonation_and_wav2vec2','mfcc_wav2vec2','double_wav2vec2']
        assert args.dataset in ['iemocap', '멀티모달 영상','msp_podcast']
        objective = getattr(args, "objective", "regression")
        if objective not in ["regression", "classification"]:
            raise ValueError("objective must be 'regression' or 'classification'.")
        if objective == "classification" and (
            args.dataset != "iemocap" or args.task != "intonation_and_wav2vec2"
        ):
            raise ValueError(
                "Categorical classification currently supports only the English "
                "IEMOCAP PCM task 'intonation_and_wav2vec2'."
            )
                
    def set_logger(self,):
        self.logger = logging.getLogger(f"iemocap_audio_train.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.logger_path)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    ## Loss function
    def MSELoss(self, pred_VADs, VADs):
        """
            pred_outs: [batch, clsNum]
            labels: [batch]
        """
        loss = nn.MSELoss()
        #print("여기까지는 들어오냐능")
        loss_val = loss(pred_VADs, VADs.float())
        #print("MSE vad : ", VADs.float())
        #print("MSE vad : ", VADs.shape)
        #print("pred ddd: ", pred_VADs)
        #print("pred : ", pred_VADs.shape)
        #print(loss_val.shape)
        #print(loss_val)
        return loss_val       
        ## 모델 저장하기
    
    #모델 저장하기
    def SaveModel(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model.bin'))

    #모델 불러오기
    def get_model(self,):
        checkpoint_path = (
            os.path.join(self.save_path, "model.bin")
            if os.path.isdir(self.save_path)
            else self.save_path
        )
        self.vad_model.load_state_dict(torch.load(checkpoint_path))
        self.vad_model.eval()
    
    # CCC score 계산            
    def concordance_correlation_coefficient(self, observed, predicted):      
        # CCC 계산
        mean_observed = np.mean(observed)
        mean_predicted = np.mean(predicted)
        #print("ob :", observed.shape)
        #print("pr : ", predicted.shape)
        covariance = np.cov(observed, predicted, bias=True)[0, 1]
        obs_variance = np.var(observed, ddof=1)
        pred_variance = np.var(predicted, ddof=1)
        CCC = 2 * covariance / (obs_variance + pred_variance + (mean_observed - mean_predicted)**2)
        return CCC
    # inference
    def CalConcordanceCorrelation(self, model, dataloader):
        model.eval()
        
        logits_v, logits_a, logits_d = [], [], []
        v, a, d = [], [], []
        
        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(dataloader,mininterval=10)):

                """Prediction"""
                dialog_id, speaker, batch_padding_tokens, batch_attention_mask, batch_audio,batch_sr, batch_vad = data

                """cuda 할당"""
                if batch_padding_tokens is not None:
                #print("여기 안 들어옴?")
                    batch_padding_tokens = batch_padding_tokens.cuda()
                    batch_attention_mask = batch_attention_mask.cuda()    
                if  batch_attention_mask is not None:
                    batch_attention_mask = batch_attention_mask.cuda()
                
                pitch_audio = None
                if isinstance(batch_audio, tuple):
                    pitch_audio = batch_audio[0].cuda()
                    batch_audio = batch_audio[1]
                if self.args.task == "mfcc_wav2vec2":
                    #print("pitch_audio : ", pitch_audio.shape)
                    pitch_audio = pitch_audio.squeeze(0).squeeze(0)
                batch_audio = batch_audio.cuda()
                batch_sr = batch_sr.cuda()
                batch_vad = batch_vad.cuda()

                #print("어텐션 마스크 : ", batch_attention_mask.is_cuda)
                #print("패딩된 토큰 : ", batch_padding_tokens.is_cuda)
                """Prediction"""  
                pred_logits = model(pitch_audio, batch_audio,batch_padding_tokens,batch_attention_mask)
                
                #print("pred_logits shape : ", pred_logits.shape)
                # 모델이 None을 반환하는 경우를 처리합니다.
                if pred_logits is None:
                    continue
                if self.args.task in ["audio_parsing_with_text","contour_image_vad"]:
                    pred_logits = pred_logits.unsqueeze(dim=0)
                if self.args.task in ["ctc_normal"]:
                    pred_logits = pred_logits[1]
                elif self.args.task in ["contour_image_and_ctc"]:
                    pred_logits = pred_logits[1].unsqueeze(dim=0)
                
                #print("pred_logits : ", pred_logits.shape)
                #print("pred_logit : ", pred_logits.shape)
                pred_v = pred_logits[:, 0].cpu().numpy()
                pred_a = pred_logits[:, 1].cpu().numpy()
                pred_d = pred_logits[:, 2].cpu().numpy()

                #print("batch_vad : ", batch_vad.shape)
                batch_v = batch_vad[:, 0].cpu().numpy()
                batch_a = batch_vad[:, 1].cpu().numpy() 
                batch_d = batch_vad[:, 2].cpu().numpy()
                
                logits_v.append(pred_v)
                logits_a.append(pred_a)
                logits_d.append(pred_d)

                v.append(batch_v) 
                a.append(batch_a)
                d.append(batch_d)  

        # Convert to NumPy arrays
        logits_v = np.concatenate(logits_v)
        logits_a = np.concatenate(logits_a)
        logits_d = np.concatenate(logits_d)
        v = np.concatenate(v)
        a = np.concatenate(a)
        d = np.concatenate(d)    

        # Calculate Concordance correlation coefficient
        ccc_V = self.concordance_correlation_coefficient(v, logits_v)
        ccc_A = self.concordance_correlation_coefficient(a, logits_a)
        ccc_d = self.concordance_correlation_coefficient(d, logits_d)

        return ccc_V, ccc_A, ccc_d
    
    def set_parameters(self,):
        """사전학습 모델 얼리기"""
        self.vad_model.freeze_prameters()
        
        """하이퍼 파라미터들"""
        self.training_epochs = self.args.epoch
        self.max_grad_norm = self.args.max_grad_norm
        self.lr = self.args.learning_rate
        self.num_training_steps = len(self.train_dataset)*self.training_epochs
        self.num_warmup_steps = len(self.train_dataset)
        self.optimizer = torch.optim.AdamW(self.vad_model.parameters(), lr=self.lr) # eps = 1e-05, weight_decay=0.01
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
    
    def _split_data(self,data): 
        train, _ = train_test_split(data, shuffle=self.args.train_shuffle,test_size=self.args.val_ratio) # ,random_state=42, 
        valid, test = train_test_split(_, shuffle=self.args.train_shuffle,test_size=self.args.test_ratio) # ,random_state=42,          
        return train, valid, test #0.8:0.1:0.1    

    def _split_iemocap_data(self, data):
        if not self.classification:
            train = data[data['name'].str.contains('Ses01|Ses02|Ses03')]
            valid = data[data['name'].str.contains('Ses04')]
            test = data[data['name'].str.contains('Ses05')]
            return train, valid, test

        required_columns = {"sample_name", "transcript", "emotion", "emotion_id", "path"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                "Categorical IEMOCAP data is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        sample_names = data["sample_name"].astype(str)
        fold_test_session = getattr(self.args, "loso_test_session", None)
        if fold_test_session is not None:
            valid_sessions = {f"Ses0{session}" for session in range(1, 6)}
            if fold_test_session not in valid_sessions:
                raise ValueError(
                    "loso_test_session must be one of Ses01, Ses02, Ses03, Ses04, or Ses05."
                )
            sessions = sample_names.str.extract(r"(Ses0[1-5])", expand=False)
            if sessions.isna().any():
                raise ValueError("IEMOCAP sample names must identify a session from Ses01 to Ses05.")
            train = data[sessions != fold_test_session]
            valid = data.iloc[0:0]
            test = data[sessions == fold_test_session]
            if train.empty or test.empty:
                raise ValueError(
                    f"LOSO fold {fold_test_session} must have non-empty train and test sessions."
                )
            return train, valid, test

        train = data[sample_names.str.match(r"Ses0[1-3]")]
        valid = data[sample_names.str.match(r"Ses04")]
        test = data[sample_names.str.match(r"Ses05")]
        if train.empty or valid.empty or test.empty:
            raise ValueError("IEMOCAP categorical data must contain sessions Ses01-Ses05.")
        return train, valid, test

    def evaluate_classification(self, model, dataloader):
        model.eval()
        true_labels = []
        predicted_labels = []
        confidences = []
        sample_ids = []

        with torch.no_grad():
            for data in tqdm(dataloader, mininterval=10):
                batch_ids, _, batch_tokens, batch_attention_mask, batch_audio, batch_sr, batch_labels = data
                batch_tokens = batch_tokens.cuda()
                batch_attention_mask = batch_attention_mask.cuda()
                pitch_audio, audio_inputs = batch_audio
                logits = model(
                    pitch_audio.cuda(),
                    audio_inputs.cuda(),
                    batch_tokens,
                    batch_attention_mask,
                )
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predictions = probabilities.max(dim=-1)
                true_labels.extend(batch_labels.tolist())
                predicted_labels.extend(predictions.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())
                sample_ids.extend(batch_ids)

        metrics = {
            "wa": accuracy_score(true_labels, predicted_labels),
            "ua": balanced_accuracy_score(true_labels, predicted_labels),
            "macro_f1": f1_score(
                true_labels, predicted_labels, average="macro", zero_division=0
            ),
        }
        return metrics, sample_ids, predicted_labels, confidences
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # 학습 시키기
    def train(self,):

        #--- 데이터셋 설정 ---#
        
        if self.args.dataset == "iemocap":
            if self.classification:
                df = pd.read_csv(self.args.text_cap_path, sep=',', encoding='utf-8')
            else:
                df = pd.read_csv(self.args.text_cap_path, sep='\t', encoding='utf-8')
            train, val, test = self._split_iemocap_data(df)
            
            #train, val, test = self._split_data(df)
        elif self.args.dataset == "msp_podcast":
            df = pd.read_csv(self.args.text_msp_path, sep=',')
            train = df[df['Split_Set'] == 'Train']#[:500]
            test = df[df['Split_Set'] == 'Test1']#[:100]
            val = df[df['Split_Set'] == 'Development']#[:100]
                            
            print(train[:5])
        
        self.train_dataset = data_loader(train,self.args)
        self.dev_dataset = data_loader(val,self.args) if not val.empty else None
        self.test_dataset = data_loader(test,self.args)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=self.args.train_shuffle,  collate_fn=self.train_dataset.collate_fn) #num_workers=4,
        self.dev_dataloader = (
            DataLoader(self.dev_dataset, batch_size=self.args.eval_batch_size, shuffle=self.args.train_shuffle,  collate_fn=self.dev_dataset.collate_fn)
            if self.dev_dataset is not None
            else None
        )
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, shuffle=self.args.train_shuffle, collate_fn=self.test_dataset.collate_fn) #num_workers=4,
        
        self.set_logger()
        self.set_parameters()
        
        #model_path = './vad_iemocap_audio_new_model/model.bin'
        #self.vad_model.load_state_dict(torch.load(model_path))

        self.logger.info("#############학습 시작#############")

        for epoch in range(self.training_epochs):
            self.vad_model.train()
            loss = 0
            
            for i_batch, data in enumerate(tqdm(self.train_dataloader,mininterval=10)):
                try:
                    dialog_id, speaker, batch_padding_tokens, batch_attention_mask, batch_audio,batch_sr, batch_vad = data

                    """cuda 할당"""
                    if batch_padding_tokens is not None:
                        #print("들어오지?")
                        #print("여기 안 들어와?")
                        batch_padding_tokens = batch_padding_tokens.cuda()
                        batch_attention_mask = batch_attention_mask.cuda()
                    if  batch_attention_mask is not None:
                        batch_attention_mask = batch_attention_mask.cuda()
                    pitch_audio = None
                    if isinstance(batch_audio, tuple):
                        pitch_audio = batch_audio[0].cuda()
                        batch_audio = batch_audio[1]
                    if self.args.task == "mfcc_wav2vec2":
                        #print("pitch_audio : ", pitch_audio.shape)
                        pitch_audio = pitch_audio.squeeze(0).squeeze(0)
                    if self.args.task == "ctc_normal":
                        batch_padding_tokens = batch_padding_tokens.cuda()
                                         
                    batch_audio = batch_audio.cuda()
                    #batch_audio = whole_audio.cuda()   
                    batch_sr = batch_sr.cuda()
                    batch_vad = batch_vad.cuda()

                    """Prediction"""
                    pred_logits = self.vad_model(pitch_audio, batch_audio,batch_padding_tokens,batch_attention_mask)
                    
                    if pred_logits is None:
                        print("Model output is None. Skipping this batch")
                        continue  # Skip this batch if model output is None
                    
                    """Loss claculation & training"""
                    if self.classification:
                        if not torch.isfinite(pred_logits).all():
                            sample_ids = dialog_id
                            pitch_is_finite = (
                                pitch_audio is not None
                                and torch.isfinite(pitch_audio).all().item()
                            )
                            audio_is_finite = torch.isfinite(batch_audio).all().item()
                            raise FloatingPointError(
                                "Model produced non-finite classification logits at "
                                f"batch {i_batch}; samples={sample_ids}; "
                                f"pitch_is_finite={pitch_is_finite}; "
                                f"audio_is_finite={audio_is_finite}."
                            )
                        if batch_vad.min().item() < 0 or batch_vad.max().item() >= pred_logits.size(-1):
                            raise ValueError(
                                "Classification labels are outside the model output range: "
                                f"samples={dialog_id}; labels={batch_vad.tolist()}; "
                                f"num_classes={pred_logits.size(-1)}."
                            )
                        loss_val = F.cross_entropy(pred_logits, batch_vad.long())
                    elif self.args.task in ['ctc_normal','contour_image_and_ctc']:
                        ctc_logits, vad_logits = pred_logits
                        # batch_size와 seq_length 가져오기
                        batch_size = ctc_logits.size(0)
                        seq_length = ctc_logits.size(1)

                        # input_lengths와 target_lengths를 시퀀스 길이와 마스크를 기반으로 계산
                        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).cuda()

                        labels_mask = batch_padding_tokens >= 0
                        target_lengths = labels_mask.sum(-1)

                        # CTC Loss 계산
                        ctc_logits = torch.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)  # CTC Loss는 log probabilities를 기대하며, 차원 변환이 필요합니다

                        # 필요한 설정을 직접 정의
                        blank_token_id = 0  # 예시로, pad_token_id가 0이라고 가정
                        ctc_loss_reduction = 'mean'  # 또는 'sum'
                        ctc_zero_infinity = True

                        with torch.backends.cudnn.flags(enabled=False):  # CUDNN 비결정성 제어
                            ctcloss = F.ctc_loss(
                                ctc_logits,
                                batch_padding_tokens.masked_select(labels_mask),
                                input_lengths,
                                target_lengths,
                                blank=blank_token_id,
                                reduction=ctc_loss_reduction,
                                zero_infinity=ctc_zero_infinity
                            )
                        
                        vadloss = self.MSELoss(vad_logits, batch_vad)
                        
                        
                        #print("ctc_loss : ", ctcloss)
                        #print("vad_loss : ", vadloss)
                        
                        loss_val = self.args.ctc_a * ctcloss + vadloss
                    else:
                        loss_val = self.MSELoss(pred_logits, batch_vad)
                    
                    #if loss_val is None:
                    #    continue  # Skip loss calculation and training if loss is None
                    
                    # NaN 값 검사
                    if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                        if self.classification:
                            raise FloatingPointError(
                                "Non-finite classification loss at "
                                f"batch {i_batch}; samples={dialog_id}; "
                                f"logits_min={pred_logits.min().item()}; "
                                f"logits_max={pred_logits.max().item()}."
                            )
                        print("NaN or Inf detected in loss_val")
                        
                    loss += loss_val.item()
                    
                    loss_val.backward(retain_graph=True)#
                    torch.nn.utils.clip_grad_norm_(self.vad_model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    #torch.cuda.empty_cache()
                    gc.collect()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        self.logger.error(f"Out of memory at batch {i_batch}: {e}")
                        torch.cuda.empty_cache()  # 캐시 비우기
                        print(f"Memory cleared at batch {i_batch}.")
                    else:
                        raise e  # 다른 오류는 다시 발생시킵니다
                except Exception as e:
                    if self.classification:
                        raise
                    print(f"An error occurred: {e}. Skipping this batch.")
                    continue  # Skip this batch if an exception occurs            
            #torch.cuda.empty_cache()
            loss = loss/len(self.train_dataloader)    
            """Dev & Test evaluation"""
            
            self.vad_model.eval()
            if self.classification:
                self.logger.info("Train Cross-Entropy Loss: {:.4f}".format(loss))
                if self.dev_dataloader is None:
                    test_metrics, _, _, _ = self.evaluate_classification(
                        self.vad_model, self.test_dataloader
                    )
                    self.SaveModel(self.vad_model, self.save_path)
                    epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": loss,
                        "test_wa": test_metrics["wa"],
                        "test_ua": test_metrics["ua"],
                        "test_macro_f1": test_metrics["macro_f1"],
                    }
                    self.epoch_metrics.append(epoch_metrics)
                    self.logger.info(
                        "Fold %s, Epoch %d/%d - Test WA: %.4f, UA: %.4f, Macro F1: %.4f",
                        self.args.loso_test_session,
                        epoch + 1,
                        self.training_epochs,
                        test_metrics["wa"],
                        test_metrics["ua"],
                        test_metrics["macro_f1"],
                    )
                    continue

                dev_metrics, _, _, _ = self.evaluate_classification(
                    self.vad_model, self.dev_dataloader
                )
                self.logger.info(
                    "Dev WA: {:.4f}, UA: {:.4f}, Macro F1: {:.4f}".format(
                        dev_metrics["wa"], dev_metrics["ua"], dev_metrics["macro_f1"]
                    )
                )
                if dev_metrics["macro_f1"] > self.best_score:
                    self.best_score = dev_metrics["macro_f1"]
                    self.SaveModel(self.vad_model, self.save_path)
                    test_metrics, _, _, _ = self.evaluate_classification(
                        self.vad_model, self.test_dataloader
                    )
                    self.logger.info(
                        "Test WA: {:.4f}, UA: {:.4f}, Macro F1: {:.4f}".format(
                            test_metrics["wa"],
                            test_metrics["ua"],
                            test_metrics["macro_f1"],
                        )
                    )
                continue

            self.logger.info("Train MSE Loss : {}".format(loss))
            dev_cccV, dev_cccA, dev_cccD = self.CalConcordanceCorrelation(self.vad_model, self.dev_dataloader)
            
            self.logger.info("\n# -- Training epoch : {:.0f} -- #".format(epoch))
            self.logger.info("\nDev avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(dev_cccV, dev_cccA, dev_cccD))
            
            dev_avg = (dev_cccV+dev_cccA+dev_cccD)/3
            
            """Best Score & Model Save"""
            curr_score = dev_avg
            if curr_score > self.best_score: 
                self.best_score = curr_score
                
                test_cccV, test_cccA, dev_cccD = self.CalConcordanceCorrelation(self.vad_model, self.test_dataloader)
                
                self.SaveModel(self.vad_model, self.save_path)     
                self.logger.info("\nTest avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(test_cccV, test_cccA, dev_cccD))
    # python cupcake_backup/iemocap_code/iemocap_audio_train.py --config /home/ryumj/cupcake_backup/configs/audio.json"

    def inference(self):
        # 데이터셋 설정
        if self.args.dataset == "iemocap":
            if self.classification:
                df = pd.read_csv(self.args.text_cap_path, sep=',', encoding='utf-8')
                _, _, test = self._split_iemocap_data(df)
            else:
                df = pd.read_csv(self.args.text_cap_path, sep='\t', encoding='utf-8')
                _, _, test = self._split_data(df)
        elif self.args.dataset == "msp_podcast":
            df = pd.read_csv(self.args.text_msp_path, sep=',', encoding='utf-8')
            _, _, test = self._split_data(df)
        self.test_dataset = data_loader(test, self.args)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.test_dataset.collate_fn)
        
        self.set_logger()
        self.set_seed(self.args.seed)
        
        self.get_model()
        
        self.logger.info("#############Inference 시작#############")
        if self.classification:
            metrics, sample_ids, predictions, confidences = self.evaluate_classification(
                self.vad_model, self.test_dataloader
            )
            self.logger.info(
                "Test WA: {:.4f}, UA: {:.4f}, Macro F1: {:.4f}".format(
                    metrics["wa"], metrics["ua"], metrics["macro_f1"]
                )
            )
            prediction_results = []
            for sample_id, label_id, confidence in zip(
                sample_ids, predictions, confidences
            ):
                result = {
                    "sample_name": sample_id,
                    "emotion": EMOTION_LABELS[label_id],
                    "emotion_vi": EMOTION_LABELS_VI[label_id],
                    "confidence": confidence,
                }
                prediction_results.append(result)
                self.logger.info(
                    "%s: %s / %s (confidence: %.4f)",
                    sample_id,
                    result["emotion"],
                    result["emotion_vi"],
                    confidence,
                )
            return prediction_results
        else:
            test_cccV, test_cccA, test_cccD = self.CalConcordanceCorrelation(self.vad_model, self.test_dataloader)
            self.logger.info("Test avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f}".format(test_cccV, test_cccA, test_cccD))
def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--inference', action='store_true')
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        args = json.load(config_file)
    args = argparse.Namespace(**args)

    emotion_trainer = Trainer(args)
    if arg_.inference:
        emotion_trainer.inference()
    else:
        emotion_trainer.train()

if __name__ == "__main__":
    main()