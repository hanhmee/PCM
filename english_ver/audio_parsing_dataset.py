from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, WhisperProcessor
from transformers import AutoTokenizer, RobertaTokenizer, AutoProcessor
#from kobert_tokenizer import KoBERTTokenizer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import numpy as np
import re
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image

import librosa
import parselmouth
from parselmouth.praat import call
from scipy.interpolate import interp1d

from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = ("happy", "neutral", "sad", "angry")
EMOTION_LABELS_VI = ("Vui", "Bình thường", "Buồn", "Tức giận")
EMOTION_TO_ID = {"hap": 0, "neu": 1, "sad": 2, "ang": 3}

# -- audio parsing class -- #
@dataclass
class Point:
  token_index: int
  time_index: int
  score: float
# -- Merge the labels -- #
@dataclass
class Segment:
  label: str
  start: int
  end: int
  score: float

  def __repr__(self):
    return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

  @property
  def length(self):
    return self.end - self.start   
class MakingAudioSegment:
    def __init__(self,):
        torch.random.manual_seed(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(torch.__version__)
        print(torchaudio.__version__)
        print(self.device)
                
    def get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t+1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return torch.tensor(trellis)
    
    def backtrack(self, trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When refering to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when refering to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.

        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t-1, j] + emission[t-1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j-1, t-1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
            if j == 0:
                break
        else:
            raise ValueError('Failed to align')
        return path[::-1]
    def merge_repeats(self, path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(Segment( self.transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
            i1 = i2
        return segments
    # Merge words
    def merge_words(self, segments, separator='|'):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = ''.join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2-1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words
    # A trick to embed the resulting audio to the generated file.
# `IPython.display.Audio` has to be the last call in a cell,
# and there should be only one call par cell.
    def saving_segment(self, ): # i는 몇번째 세그먼트인지. 
        audio_segments = []
        ratio = self.waveform.size(1) / (self.trellis.size(0) - 1)
        for i in range(len(self.word_segments)):
            word = self.word_segments[i]
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)
            #filename = f"/content/sample_data/{i}_{word.label}.wav"
            #torchaudio.save(filename, self.waveform[:, x0:x1], self.bundle.sample_rate)
            #print(f"{word.label} ({word.score:.2f}): {x0 / self.bundle.sample_rate:.3f} - {x1 / self.bundle.sample_rate:.3f} sec")
            audio_segments.append(self.waveform[:, x0:x1])
        
        return audio_segments

    def processing(self,waveform, transcript, sr):
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()
        
        self.waveform = waveform
        self.transcript = transcript
        self.sr = sr
        
        #print("오류가 왜 날까2222 : ", type(self.waveform))
        
        with torch.inference_mode():
            #self.waveform = torch.from_numpy(self.waveform).unsqueeze(-1).transpose(0,1)
            self.waveform = torch.tensor(self.waveform)
            
            #waveform, _ = torchaudio.load(SPEECH_FILE)
            emissions, _ = self.model(self.waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()
        
        
        dictionary  = {c: i for i, c in enumerate(self.labels)}
        tokens = [dictionary[c] for c in self.transcript]
        #print(list(zip(self.transcript, tokens)))
        
        self.trellis = self.get_trellis(emission, tokens)
        path = self.backtrack(self.trellis, emission, tokens)
        segments = self.merge_repeats(path)
        self.word_segments = self.merge_words(segments)
        
        audio_segments = self.saving_segment()
        
        return audio_segments
    def get_segments(self,):
        return self.word_segments

# -- audio pitch class -- #
class Intonation_features:
    def __init__(self,sound):
        #self.args = args
        
        self.sound = sound 
        self.pitch_tiers = None
        self.total_duration = None
        self.pitch_point = []
        self.time_point = []
        
        self.pd = []
        self.pt = []
        self.ps = []
        self.pr = []
        
    def get_pitch_tiers(self,):
        manipulation = call(self.sound, "To Manipulation", 0.01, 75, 600)
        self.pitch_tier = call(manipulation, "Extract pitch tier")
        return self.pitch_tier
        
    def stylize_pitch(self,):
        if self.pitch_tier is not None:
            call(self.pitch_tier, "Stylize...",2.0,"semitones")
            tmp_pitch_point = self.pitch_point
            tmp_time_point = self.time_point
            self.set_time_and_pitch_point()
            if len(self.pitch_point) == 0:
                self.pitch_point = tmp_pitch_point
                self.time_point = tmp_time_point 
        else:
            print("pitch_tier is None")
            return 
        #self.pitch_tier.save("./Pitch_Tier/tmp.PitchTier")
        
    def set_total_duration(self,):
        # 정규 표현식을 사용하여 Total duration 값을 추출

        total_duration_match = re.search(r'Total duration: (\d+(\.\d+)?) seconds', str(self.pitch_tier))
        # Total duration 값이 존재하는지 확인 후 출력
        if total_duration_match:
            self.total_duration = float(total_duration_match.group(1))
        else:
            print(str(self.pitch_tier))
            print("Total duration not found.")
            
    def set_time_and_pitch_point(self,):
        self.pitch_tier.save("./tmp.PitchTier")#("./Pitch_Tier/tmp.PitchTier")
        r_file = open("./tmp.PitchTier")#("./Pitch_Tier/tmp.PitchTier",'r')
        
        self.pitch_point = []
        self.time_point = []
        while True:
            line = r_file.readline()
            if not line:
                break

            if 'number' in line:
                value = re.sub(r'[^0-9^.]', '', line)
                # 불필요한 . 제거 (뒤쪽 . 하나 제거)
                if value.count('.') > 1:
                    parts = value.split('.')
                    value = parts[0] + ''.join(parts[1:])
                if value != '':
                    self.time_point.append(round(float(value), 4))
            elif 'value' in line:
                value = re.sub(r'[^0-9^.]', '', line)
                # 불필요한 . 제거 (뒤쪽 . 하나 제거)
                if value.count('.') > 1:
                    parts = value.split('.')
                    value = parts[0] + ''.join(parts[1:])
                if value != '':
                    self.pitch_point.append(round(float(value), 4))
                
        if len(self.pitch_point)==0:
            print("왜 ????????????? 말이 돼??????????")
            while True:
                line = r_file.readline()
                if not line:
                    break
                print(line)
        r_file.close()
    
    #피치 포인트들 간의 거리        
    def extract_pd(self,):
        for i in range(len(self.pitch_point) - 1):
            point1 = (self.time_point[i], self.pitch_point[i])
            point2 = (self.time_point[i+1], self.pitch_point[i+1])
            #euclidean_distance
            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            self.pd.append(round(distance,4))

    #피치 포인트 이동시간
    def extract_pt(self,):
        for i in range(len(self.time_point)-1):
            gap = self.time_point[i+1]-self.time_point[i]
            percent = round((gap/self.total_duration),4)
            self.pt.append(percent)
    
    #피치 포인트들 간의 변위
    def extract_pr(self,):
        if len(self.pitch_point) == 0:
            return
        total = max(self.pitch_point) - min(self.pitch_point)
        for i in range(1, len(self.pitch_point)):
            delta_y = abs(self.pitch_point[i] - self.pitch_point[i-1])
            
            r = round(delta_y / total,4)
            self.pr.append(r) 
    
    #피치 포인트들 간의 기울기
    def extract_ps(self,):
        for i in range(1, len(self.time_point)):
            delta_x = self.time_point[i] - self.time_point[i-1]
            delta_y = self.pitch_point[i] - self.pitch_point[i-1]
            
            if delta_x == 0:
                # 기울기를 정의할 수 없는 경우 (분모가 0인 경우)는 None을 추가
                print("이게 있으면 안되는뎀.")
                self.ps.append(None)
            else:
                slope = delta_y / delta_x
                self.ps.append(round(slope,4))
    def get_features(self,):
        
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point() # 추가됨 너무 짧은 문장들 (피치 티어 수가 원래 적은 문장) 때문에.
        self.stylize_pitch()
        #self.set_time_and_pitch_point()
        self.set_total_duration()
        feature = []
        #if self.args.features == 'intonation_with_text':
        self.extract_pd()
        self.extract_pt()
        self.extract_ps()
        self.extract_pr()
        
        #feature = self.pd + self.pt + self.pr + self.ps
        for t,d,s,r in zip(self.pt, self.pd, self.ps,self.pr):
#            tdsr = t * d * s * r
#            if tdsr < 0:
#                f = np.sqrt(abs(tdsr))
#                f = round(-1 * f,2)
#            else:
#                f = np.sqrt(abs(tdsr))
#                f = round(f, 2)
#            feature.append(f)
            td, rs =  t * d , r * s
            if s < 0:
                rs = np.sqrt(abs(rs))
                rs = round(-1 * rs, 2)
            td = np.sqrt(td)
            feature.append(td)#(t)
            feature.append(rs)#d)
            #feature.append(s)
            #feature.append(r)
        
        return feature
    def get_pitchs(self,):
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point() # 추가됨 너무 짧은 문장들 (피치 티어 수가 원래 적은 문장) 때문에.
        return self.pitch_point, self.time_point

    def get_intensity(self,):
        intensity = call(self.sound, "To Intensity...", 100.0, 0.0)
        return intensity.values[0]


def z_score_normalization(pitch_values):
    pitch_values = np.asarray(pitch_values, dtype=np.float32)
    finite_mask = np.isfinite(pitch_values)
    if not finite_mask.any():
        return np.zeros_like(pitch_values)

    finite_values = pitch_values[finite_mask]
    mean = finite_values.mean()
    std = finite_values.std()
    if not np.isfinite(std) or std < 1e-6:
        normalized_pitch = np.zeros_like(pitch_values)
    else:
        normalized_pitch = np.zeros_like(pitch_values)
        normalized_pitch[finite_mask] = (pitch_values[finite_mask] - mean) / std
    return normalized_pitch
    
    
# -- dataloader class -- #
class data_loader(Dataset):
    def __init__(self, data, args):
        
        self.args = args
        
        # -- for audio -- #
        self.audio_model_name = self.args.audio_model_name # "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        if args.task in ["audio_parsing"]:
            self.audio_model = Wav2Vec2Model.from_pretrained(self.audio_model_name, mask_feature_length=10)#.cuda()
            self.audio_tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        
        # -- for text -- #
        self.tokenizer = RobertaTokenizer.from_pretrained(self.args.tokenizer,padding_side='left')

        # -- for image -- #
        self.image_path = self.args.image_path
        self.transform = transform = transforms.Compose([        
            transforms.ToTensor(),
        ])#transforms.Resize((1200, 3000)),  # 이미지 크기 조정
        
        self.saved_data = []
        self.wav_path_list = []
        self.audio_path = self.args.audio_path

        self.classification = getattr(self.args, "objective", "regression") == "classification"
        audio_paths = None
        if self.args.dataset == "msp_podcast":
            name = list(data['FileName'])
            V = list(data['V'])
            A = list(data['A'])
            D = list(data['D'])
        elif self.args.dataset == "iemocap":
            if self.classification:
                required_columns = {"sample_name", "transcript", "emotion", "emotion_id", "path"}
                missing_columns = required_columns.difference(data.columns)
                if missing_columns:
                    raise ValueError(
                        "Categorical IEMOCAP data is missing required columns: "
                        + ", ".join(sorted(missing_columns))
                    )
                name = list(data["sample_name"])
                texts = list(data["transcript"])
                emotion_ids = data["emotion_id"].astype(int).tolist()
                audio_paths = list(data["path"])
                expected_ids = data["emotion"].map(EMOTION_TO_ID)
                if expected_ids.isna().any() or not np.array_equal(
                    expected_ids.to_numpy(dtype=int), np.asarray(emotion_ids)
                ):
                    raise ValueError(
                        "IEMOCAP emotion labels must match emotion_id mapping "
                        "{hap: 0, neu: 1, sad: 2, ang: 3}."
                    )
            else:
                name = list(data['name'])
                texts = list(data['sentence'])
                V = list(data['V'])
                A = list(data['A'])
                D = list(data['D'])

        if self.args.task in ['normal','temporal_normal']:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                whole_audio = waveform
                self.saved_data.append([V[i],A[i],D[i],whole_audio,re_sr])
                
        elif self.args.task in ['msp_normal']:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i]
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                whole_audio = waveform
                if waveform is None:
                    continue
                self.saved_data.append([V[i],A[i],D[i],whole_audio,re_sr])
        
        elif self.args.task in ["normal_with_text","ctc_normal"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                whole_audio = waveform
                
                dialog_id = name[i][:-5]
                speaker = name[-4]            
                sentence = texts[i]
                
                self.saved_data.append([dialog_id, sentence, V[i],A[i],D[i],whole_audio,re_sr,speaker])
                
        elif self.args.task in [ "contour_image_and_ctc", "contour_image_vad"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                png_path = self.image_path + name[i]+'.png' 
                image = Image.open(png_path).convert('RGB')
                image = self.transform(image)
                sentence = texts[i]
                
                self.saved_data.append([V[i],A[i],D[i],image,sentence,waveform,re_sr])
                
        elif self.args.task == "contour_image_and_text":
            for i in tqdm(range((len(data))),mininterval=10):
                
                png_path = self.image_path + name[i]+'.png' 
                image = Image.open(png_path).convert('RGB')
                image = self.transform(image)
                sentence = texts[i]
                
                self.saved_data.append([V[i],A[i],D[i],image,text])
        elif self.args.task in ["audio_parsing","audio_parsing_probing"]:
            segmentation = MakingAudioSegment()
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)

                text = texts[i]
                text = text.replace("\n", "")
                text = text.replace("\t", "")
                text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]','', text)
                text = re.sub(r'[0-9]', '', text)
                text = text.upper()
                text = text.lstrip()
                text = text.rstrip()
                text = text.replace(" ","|")
                
                if self.args.subtask == "parsing_first": #  1번 구조
                    audio_segments = segmentation.processing(waveform,text,re_sr) #원복해야됨
                    whole_audio= []
                    for j, a in enumerate(audio_segments):
                        if len(a[0]) > 10000:
                            linear = torch.nn.Linear(len(a[0]), 10000) # 2000보다 짧으면 리니어 레이어 통과가 안 됨
                            passed_a = linear(a[0].clone().detach())
                        else:
                            passed_a = a[0].clone()    
                        padded_a = torch.cat((torch.tensor(passed_a), torch.zeros(10000-len(passed_a))))  # 패딩 추가
                        whole_audio.append(padded_a)
                    whole_audio = torch.stack(whole_audio, dim=0)
                    whole_audio.detach().numpy()
                    self.saved_data.append([V[i],A[i],D[i],whole_audio,re_sr])
                elif self.args.subtask == "embedding_first": # 2번 구조
                    audio_segments = segmentation.processing(waveform,text,re_sr) #원복해야됨
                    whole_audio= []
                    for a in audio_segments:
                        current_size = a.size()
                        padded_a = F.pad(a, (0, 1000 - current_size[0]), mode='constant', value=0)
                        input = self.processor(audio=padded_a, sample_rate=16000, return_tensors="pt",padding=True).input_values#.to(device)
                        #print("input.shape", input.shape)
                        logits = self.audio_model(input.squeeze(dim=0),output_hidden_states=True).hidden_states[12]#
                        #print("last_hiddne_state size", logits.shape)
                        word_input = torch.mean(logits.squeeze(dim=0), dim=0, keepdim=True)
                        #print("word_input_size", word_input.shape)
                        whole_audio.append(word_input)    

                    whole_audio = torch.stack(whole_audio, dim=0).squeeze(dim=1) 
                    #print("whole_audio.shape", whole_audio.shape)   
                    whole_audio = torch.mean(whole_audio, dim=0).unsqueeze(dim=0)
                    print("whole_audio.shape", whole_audio.shape)  
                    self.saved_data.append([V[i],A[i],D[i],whole_audio,re_sr]) #waveform#concat_audio
                
                elif self.args.subtask == "probing": # 2번 구조
                    audio_segments = segmentation.processing(waveform,text,re_sr) #원복해야됨
                    decoded_sentence = ''
                    for a in audio_segments:
                        current_size = a.size()
                        padded_a = F.pad(a, (0, 1000 - current_size[0]), mode='constant', value=0)
                        input = self.processor(audio=padded_a, sample_rate=16000, return_tensors="pt",padding=True).input_values#.to(device)
                        #print("input.shape", input.shape)
                        logits = self.audio_model(input.squeeze(dim=0),output_hidden_states=True).hidden_states[12]#
                        # 로그 확률을 인덱스로 변환
                        predicted_ids = torch.argmax(logits, dim=-1)
                        # 인덱스를 문자로 변환
                        transcription = self.audio_tokenizer.batch_decode(predicted_ids)[0]
                        print("Transcription:", transcription)
                        decoded_sentence += transcription
                                            
                    self.saved_data.append([V[i],A[i],D[i],decoded_sentence,re_sr]) #waveform#concat_audio
        
        elif self.args.task == "intonation_contour":
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                #waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_feature = intonation.get_features()
                
                # padding and truncation
                max_length = 70
                
                if len(pitch_feature) < max_length:
                    need = max_length - len(pitch_feature)
                    pad = [0] * need 
                    pitch_feature += pad
                elif len(pitch_feature) >max_length:
                    pitch_feature = pitch_feature[:max_length]
                self.saved_data.append([V[i],A[i],D[i],pitch_feature,sr])
                        
        elif self.args.task in ["audio_parsing_with_text", "audio_parsing_with_text_concat"]:
            segmentation = MakingAudioSegment()
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)

                text = texts[i]
                text = text.replace("\n", "")
                text = text.replace("\t", "")
                text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]','', text)
                text = re.sub(r'[0-9]', '', text)
                text = text.upper()
                text = text.lstrip()
                text = text.rstrip()
                text = text.replace(" ","|")
                
                if self.args.subtask == "linear_concat":
                    audio_segments = segmentation.processing(waveform,text,re_sr) #원복해야됨
                    whole_audio= []
                    for j, a in enumerate(audio_segments):
                        if len(a[0]) > 10000:
                            linear = torch.nn.Linear(len(a[0]), 10000) # 2000보다 짧으면 리니어 레이어 통과가 안 됨
                            passed_a = linear(a[0].clone().detach())
                        else:
                            passed_a = a[0].clone()    
                        padded_a = torch.cat((torch.tensor(passed_a), torch.zeros(10000-len(passed_a))))  # 패딩 추가
                        whole_audio.append(padded_a)
                    whole_audio = torch.stack(whole_audio, dim=0)
                    whole_audio.detach().numpy()
                    
                    dialog_id = name[i][:-5]
                    speaker = name[-4]            
                    sentence = texts[i]
                    
                    self.saved_data.append([dialog_id, sentence, V[i],A[i],D[i],whole_audio,re_sr,speaker])  
                elif self.args.subtask == "parsing_first": #  1번 구조
                    audio_segments = segmentation.processing(waveform,text,re_sr) #원복해야됨
                    whole_audio= []
                    for j, a in enumerate(audio_segments):
                        if len(a[0]) > 10000:
                            linear = torch.nn.Linear(len(a[0]), 10000) # 2000보다 짧으면 리니어 레이어 통과가 안 됨
                            passed_a = linear(a[0].clone().detach())
                        else:
                            passed_a = a[0].clone()    
                        padded_a = torch.cat((torch.tensor(passed_a), torch.zeros(10000-len(passed_a))))  # 패딩 추가
                        whole_audio.append(padded_a)
                    whole_audio = torch.stack(whole_audio, dim=0)
                    whole_audio.detach().numpy()
                    
                    segments = segmentation.get_segments()
                    word_list = []
                    for s in segments:
                        word_list.append(s.label.lower())
                    dialog_id = name[i][:-5]
                    speaker = name[-4]            
                    
                    self.saved_data.append([dialog_id, word_list, V[i],A[i],D[i],whole_audio,re_sr,speaker])  
                elif self.args.subtask == "embedding_first": # 2번 구조
                    
                    audio_segments = segmentation.processing(waveform,text,re_sr) #원복해야됨
                    whole_audio= []
                    for a in audio_segments:
                        current_size = a.size()
                        padded_a = F.pad(a, (0, 1000 - current_size[0]), mode='constant', value=0)
                        input = self.processor(audio=padded_a, sample_rate=16000, return_tensors="pt",padding=True).input_values#.to(device)
                        logits = self.audio_model(input.squeeze(dim=0),output_hidden_states=True).hidden_states[12]#
                        word_input = torch.mean(logits.squeeze(dim=0), dim=0, keepdim=True)
                        whole_audio.append(word_input)    

                    whole_audio = torch.stack(whole_audio, dim=0).squeeze(dim=1) 
                    #print("whole_audio.shape", whole_audio.shape)   
                    whole_audio = torch.mean(whole_audio, dim=0).unsqueeze(dim=0)
                    print("whole_audio.shape", whole_audio.shape)  
                    dialog_id = name[i][:-5]
                    speaker = name[-4]            
                    sentence = texts[i]
                    
                    self.saved_data.append([dialog_id, sentence, V[i],A[i],D[i],whole_audio,re_sr,speaker])            
        
        elif self.args.task in ["intonation_contour_with_text"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                #waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_tiers = intonation.get_pitchs()
                
            
                linear = torch.nn.Linear(len(pitch_tiers), 1024)
                pitch_features = linear(torch.tensor(pitch_tiers))
                
                dialog_id = name[i][:-5]
                speaker = name[-4]            
                sentence = texts[i]
                
                self.saved_data.append([dialog_id, sentence, V[i],A[i],D[i],pitch_features.unsqueeze(0),sr,speaker])
        
        elif self.args.task in ["intonation_and_wav2vec2","temporal_intonation_and_wav2vec2"]:
            for i in tqdm(range((len(data))),mininterval=10):
                if self.classification:
                    wav_path = self._resolve_classification_audio_path(audio_paths[i])
                else:
                    wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                if self.classification and waveform is None:
                    raise RuntimeError(f"Could not load classification audio file: {wav_path}")
                if self.classification and not np.isfinite(waveform).all():
                    raise FloatingPointError(
                        f"Non-finite audio samples for IEMOCAP utterance {name[i]}: {wav_path}"
                    )
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_tiers, time_points = intonation.get_pitchs()
                
                if not time_points or not pitch_tiers:
                    print("pitch list empty")
                    continue

                if self.classification:
                    pitch_tiers = np.asarray(pitch_tiers, dtype=np.float32)
                    finite_pitch = np.isfinite(pitch_tiers)
                    pitch_tiers = pitch_tiers[finite_pitch]
                    if pitch_tiers.size == 0:
                        print(f"pitch list contains no finite values: {name[i]}")
                        continue
                
                if self.args.normalized == "ok":
                    normalized_pitch = z_score_normalization(pitch_tiers)
                else:
                    normalized_pitch = pitch_tiers       
                
                if self.classification:
                    pitch_indexes = np.arange(len(normalized_pitch))
                    target_indexes = np.linspace(
                        0, len(normalized_pitch) - 1, self.args.hidden_size
                    )
                    pitch_features = torch.tensor(
                        np.interp(target_indexes, pitch_indexes, normalized_pitch),
                        dtype=torch.float32,
                    )
                    if not torch.isfinite(pitch_features).all():
                        raise FloatingPointError(
                            f"Non-finite pitch features for IEMOCAP utterance {name[i]}"
                        )
                elif self.args.embedding =='linear':  
                    linear = torch.nn.Linear(len(normalized_pitch), self.args.hidden_size)
                    pitch_features = linear(torch.tensor(normalized_pitch, dtype=torch.float32))
                elif self.args.embedding == 'cnn':
                    pitch_features = torch.tensor(normalized_pitch, dtype=torch.float32)
                elif self.args.embedding == 'interpolated': #이스터에그
                    ### 선형 보간 함수 생성
                    interpolation_function = interp1d(time_points, pitch_tiers, kind='linear') 
                    # 등간격의 시간 포인트 생성       
                    time_uniform = np.arange(min(time_points), max(time_points) - 1e-6 , 0.001)
                    # 보간된 피치 값 계산
                    pitch_interpolated = interpolation_function(time_uniform)
                    pitch_features = torch.tensor(pitch_interpolated).unsqueeze(-1)
                
                dialog_id = name[i][:-5]
                sentence = texts[i]
                if self.classification:
                    self.saved_data.append(
                        [name[i], sentence, emotion_ids[i], pitch_features, waveform, re_sr]
                    )
                else:
                    speaker = name[-4]
                    self.saved_data.append(
                        [dialog_id, sentence, V[i], A[i], D[i], pitch_features, waveform, sr, speaker]
                    )
            if self.args.normalized == "ok":
                print("pitch was normalized]")
            else:
                normalized_pitch = pitch_tiers
                print("wasnt normalized")  
                
        elif self.args.task in ["interpolated_intonation_and_wav2vec2"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_tiers, time_points = intonation.get_pitchs()
                
                
                if not time_points or not pitch_tiers:
                    print("pitch list empty")
                    continue
                # 선형 보간 함수 생성
                interpolation_function = interp1d(time_points, pitch_tiers, kind='linear') 
                # 등간격의 시간 포인트 생성       
                time_uniform = np.arange(min(time_points), max(time_points) - 1e-6 , 0.01)
                # 보간된 피치 값 계산
                pitch_interpolated = interpolation_function(time_uniform)
                
                
                linear = torch.nn.Linear(len(pitch_interpolated), self.args.hidden_size)
                pitch_features = linear(torch.tensor(pitch_interpolated, dtype=torch.float32))
                
                dialog_id = name[i][:-5]
                speaker = name[-4]            
                sentence = texts[i]
                
                self.saved_data.append([dialog_id, sentence, V[i],A[i],D[i],pitch_features,waveform,sr,speaker]) 
    
        elif self.args.task in ["msp_intonation_and_wav2vec2"]:
            for i in tqdm(range((len(data))),mininterval=10):
                
                wav_path = self.audio_path +'/'+ name[i]
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                if waveform is None:
                    continue
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_tiers = intonation.get_pitchs()
                if self.args.normalized == "ok":
                    normalized_pitch = z_score_normalization(pitch_tiers)
                else:
                    normalized_pitch = pitch_tiers
                               
                if self.args.embedding == "linear":
                    # linear embedding   
                    linear = torch.nn.Linear(len(normalized_pitch), self.args.hidden_size)
                    pitch_features = linear(torch.tensor(normalized_pitch, dtype=torch.float32))
                elif self.args.embedding == 'cnn':
                    # cnn embedding 
                    pitch_features = torch.tensor(normalized_pitch).unsqueeze(-1)
                    pitch_features = pitch_features.mean(axis=0) # msp has two channels
                
                self.saved_data.append([V[i],A[i],D[i],pitch_features,waveform,sr]) 
            if self.args.normalized == "ok":
                print("pitch was normalized!!!!")
            else:
                normalized_pitch = pitch_tiers
                print("wasnt normalized")
        
        elif self.args.task in ["mfcc_wav2vec2"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                # Extract MFCC features
                mfcc_features = self.extract_mfcc(waveform, re_sr)
                
                dialog_id = name[i][:-5]
                speaker = name[-4]            
                sentence = texts[i]
                
                self.saved_data.append([dialog_id, sentence, V[i],A[i],D[i],mfcc_features,waveform,sr,speaker])     
        
        elif self.args.task in ["double_wav2vec2"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                whole_audio = waveform
                self.saved_data.append([V[i],A[i],D[i],whole_audio,re_sr])    
        # -- masked features -- #             
        elif self.args.task =="masked_intonation":
            for i in tqdm(range((len(data))),mininterval=20):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_tiers = intonation.get_pitchs()
                
                #linear = torch.nn.Linear(len(pitch_tiers), 1024)
                #pitch_features = linear(torch.tensor(pitch_tiers))
                
                self.saved_data.append([V[i],A[i],D[i],torch.tensor(pitch_tiers),waveform,sr])  
                
        elif self.args.task in ["masked_mfcc"]:
            for i in tqdm(range((len(data))),mininterval=10):
                wav_path = self.audio_path +'/'+ name[i][:-5]+'/'+ name[i]+'.wav' 
                sr = 16000
                desired_duration = 10.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                # Extract MFCC features
                mfcc_features = self.extract_mfcc(waveform, re_sr)
                
                dialog_id = name[i][:-5]
                speaker = name[-4]            
                sentence = texts[i]
                
                self.saved_data.append([V[i],A[i],D[i],mfcc_features,waveform,sr])    
                    
        print(len(self.saved_data))
        print(self.saved_data[:2])
        
    def extract_mfcc(self, waveform, sample_rate, num_mfcc=13):
        # Check if waveform is a tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        
        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=num_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )
        
        mfcc_features = mfcc_transform(waveform)
        # Transpose MFCC to (batch_size, seq_len, num_mfcc)
        mfcc_features = mfcc_features.transpose(1, 2)
        
        return mfcc_features
    
    def __len__(self):
        return len(self.saved_data)
    
    def load_wav(self,wav_path,desired_duration=3.0,resample_sr=16000):
        try:
            waveform, sample_rate = torchaudio.load(wav_path)
        except Exception as e:
            print(f"An error occurred at wav_path = {wav_path}: {str(e)}")
            return None, None        
        
        """
	    # Calculate the duration of the audio
        audio_duration = waveform.size(1) / sample_rate
        
        # Check if the audio duration is longer than the desired duration
        if audio_duration > desired_duration:
            # Calculate the number of samples corresponding to 3 seconds
            desired_samples = int(desired_duration * sample_rate)

            # Get the last 5 seconds of the waveform
            waveform = waveform[:, -desired_samples:]
        else:
            # If the audio is shorter than 3 seconds, use it as is
            waveform = waveform        
        #print("waveform type :", type(waveform))
        """
        try:
            waveform = librosa.resample(y=np.array(waveform), orig_sr=sample_rate, target_sr=resample_sr)
            
        except Exception as e:
            print(f"An error occurred during resampling: {str(e)}")
            return None, None        
        #waveform = librosa.resample(y=np.array(waveform), orig_sr=sample_rate, target_sr=resample_sr)

        #print('waveform 사이즈', torch.tensor(waveform).shape)
        return waveform, resample_sr		

    def __getitem__(self, idx):
        return self.saved_data[idx]

    def _resolve_classification_audio_path(self, audio_path):
        path = Path(audio_path)
        if not path.is_absolute():
            path = Path(self.audio_path) / path
            if path.is_file():
                return str(path)

            parts = Path(audio_path).parts
            if parts and parts[0].lower() == "sentences_wav":
                path = Path(self.audio_path).joinpath(*parts[1:])
        if not path.is_file():
            raise FileNotFoundError(f"Classification audio file does not exist: {path}")
        return str(path)
    
    def collate_fn(self, batch):
        
        #입력 : 배치 단위 데이터
        #출력 : torch.tensor(batch_V), torch.tensor(batch_A), torch.tensor(batch_D), audio_input_values
        
        batch_dialog_id, batch_text_input,batch_audio,batch_sr, batch_vad, batch_speaker, batch_raw_audio = [],[],[],[],[],[],[]
        batch_image = []
        
        if self.args.task in ["normal","temporal_normal","audio_parsing","intonation_contour","double_wav2vec2","msp_normal"]:
            for i,row in enumerate(batch):
                V, A, D, waveform, sr  = row           
                batch_vad.append([V,A,D])
                batch_audio.append(waveform)
                batch_sr.append(sr)
        elif self.args.task in ['contour_image_and_ctc','contour_image_vad']:
            for i, row in enumerate(batch):
                V,A,D,image,text,waveform,sr = row
                batch_vad.append([V,A,D])
                batch_audio.append(waveform)
                batch_text_input.append(text)
                batch_image.append(image)
                batch_sr.append(sr)
        elif self.args.task in ["audio_parsing_probing"]:
            for i,row in enumerate(batch):
                V, A, D, text, sr  = row           
                batch_vad.append([V,A,D])
                batch_text_input.append(text)
                batch_sr.append(sr)
        elif self.args.task in ["ctc_normal","normal_with_text","audio_parsing_with_text","audio_parsing_with_text_concat","intonation_contour_with_text"]:
            for i,row in enumerate(batch):
                dialog_id, text, V, A, D, waveform, sr,speaker  = row
                batch_dialog_id.append(dialog_id)
                batch_text_input.append(text)            
                batch_vad.append([V,A,D])
                batch_audio.append(waveform)
                batch_sr.append(sr)
                batch_speaker.append(speaker)
        elif self.args.task in ['intonation_and_wav2vec2','temporal_intonation_and_wav2vec2','mfcc_wav2vec2','interpolated_intonation_and_wav2vec2']:
            if self.classification:
                for row in batch:
                    dialog_id, text, emotion_id, pitch, raw_audio, sr = row
                    batch_dialog_id.append(dialog_id)
                    batch_text_input.append(text)
                    batch_vad.append(emotion_id)
                    batch_audio.append(pitch)
                    batch_raw_audio.append(raw_audio)
                    batch_sr.append(sr)
            else:
                for i,row in enumerate(batch):
                    dialog_id, text, V, A, D, pitchs, raw_audio, sr, speaker = row
                    batch_dialog_id.append(dialog_id)
                    batch_text_input.append(text)            
                    batch_vad.append([V,A,D])
                    batch_audio.append(pitchs)
                    batch_raw_audio.append(raw_audio)
                    batch_sr.append(sr)
        elif self.args.task in ['masked_intonation','masked_mfcc']:
            for i, row in enumerate(batch):
                V,A,D,pitchs,waveform,sr = row
                batch_vad.append([V,A,D])
                batch_audio.append(pitchs) # pitch or mfcc
                batch_raw_audio.append(waveform)
                batch_sr.append(sr)
        elif self.args.task in ['contour_image_and_text']:
            for i, row in enumerate(batch):
                V,A,D, image = row
                batch_vad.append([V,A,D])
                batch_audio.append(image)
        elif self.args.task in ['msp_intonation_and_wav2vec2']:
            for i, row in enumerate(batch):
                V,A,D,pitch,waveform,sr = row
                batch_vad.append([V,A,D])
                batch_audio.append(pitch)   
                batch_raw_audio.append(waveform)
                batch_sr.append(sr)
        
        if self.args.task in ["normal","msp_normal"]:
            batch_audio = [audio.squeeze().flatten() for audio in batch_audio]
            #audio_input_values = self.processor(padding=True,audio=batch_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            #audio_input_values = self.processor(audio=batch_audio,sampling_rate=16000,return_tensors="pt").input_features # whisper
            #print("audio_input_value : ", audio_input_values.shape)
            audio_input_values = batch_audio
            audio_input_values = torch.tensor(audio_input_values)#.squeeze()
            
            return None,None,None,None,audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)

        if self.args.task in ["temporal_normal"]:
            batch_audio = [audio.squeeze().flatten() for audio in batch_audio]
            
            max_length = 96000
            
            # 1. Truncation: 각 음성 데이터를 max_length에 맞게 자르기
            truncated_audio = [audio[:max_length] if len(audio) > max_length else audio for audio in batch_audio]
            
            # 2. Padding: max_length보다 짧은 경우 0으로 채우기
            padded_audio = [
                torch.cat([torch.tensor(audio), torch.zeros(max_length - len(audio))]) if len(audio) < max_length else torch.tensor(audio)
                for audio in truncated_audio
            ]
            
            # 3. Attention mask 생성: 1은 유효 데이터, 0은 패딩 부분
            attention_masks = [
                torch.cat([torch.ones(audio.shape[0]), torch.zeros(max_length - audio.shape[0])]) if audio.shape[0] < max_length else torch.ones(max_length)
                for audio in truncated_audio
            ]


            # 4. Tensor 변환
            audio_input_values = torch.stack(padded_audio)
            attention_masks = torch.stack(attention_masks)
            
            
            return None,None,None,attention_masks,audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)
        
        elif self.args.task == "audio_parsing":
            batch_audio = [audio.squeeze().flatten() for audio in batch_audio]
            audio_input_values = self.processor(padding=True,audio=batch_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            audio_input_values = torch.tensor(audio_input_values).squeeze(dim=0)
            return None,None,None,None,audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)
        
        elif self.args.task == "intonation_contour":
            audio_input_values = torch.tensor(batch_audio)  
            return None,None,None,None,audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)            
        
        elif self.args.task == "normal_with_text":
            batch_audio = [audio.flatten() for audio in batch_audio]
            batch_audio = list(batch_audio)
            audio_input_values = self.processor(audio=batch_audio,padding=True,sampling_rate=16000,return_tensors="pt").input_values #truncation=True,"max_length"max_length=32000, 
            audio_input_values = torch.tensor(audio_input_values).squeeze(dim=0)
            
            inputs =  self.tokenizer(batch_text_input, padding=True,truncation=True, return_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return batch_dialog_id,batch_speaker,batch_padding_token, batch_padding_attention_mask, audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)
        
        elif self.args.task == "ctc_normal":
            batch_audio = [audio.flatten() for audio in batch_audio]
            batch_audio = list(batch_audio)
            audio_input_values = self.processor(audio=batch_audio,padding=True,sampling_rate=16000,return_tensors="pt").input_values #truncation=True,"max_length"max_length=32000, 
            audio_input_values = torch.tensor(audio_input_values)
            #print("audio_input_values : ", audio_input_values.shape)
            
            batch_text_labels =  self.processor(text=batch_text_input, return_tensors="pt").input_ids
            return batch_dialog_id,batch_speaker,batch_text_labels, torch.tensor([0]), audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)
        elif self.args.task in ['contour_image_and_ctc','contour_image_vad']:
            batch_audio = [audio.flatten() for audio in batch_audio]
            batch_audio = list(batch_audio)
            audio_input_values = self.processor(audio=batch_audio,padding=True,sampling_rate=16000,return_tensors="pt").input_values #truncation=True,"max_length"max_length=32000, 
            audio_input_values = torch.tensor(audio_input_values)
            #print("audio_input_values : ", audio_input_values.shape)
            
            batch_text_labels =  self.processor(text=batch_text_input, return_tensors="pt").input_ids
            
            batch_image = torch.stack(batch_image)
            return None, None, batch_text_labels, batch_image, audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)
            
        elif self.args.task == "audio_parsing_with_text":
            #batch_audio = [audio.flatten() for audio in batch_audio]
            #batch_audio = list(batch_audio)
            audio_input_values = self.processor(audio=batch_audio,padding=True,sampling_rate=16000,return_tensors="pt").input_values #truncation=True,"max_length"max_length=32000, 
            audio_input_values = torch.tensor(audio_input_values).squeeze(dim=0).squeeze(dim=0) 
            #print("audio_input_values shape : ", audio_input_values.shape)
            
            #print(batch_text_input)
            #print("batch_text_input",len(batch_text_input))
 
            inputs =  self.tokenizer(batch_text_input[0], padding=True,truncation=True, return_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return batch_dialog_id,batch_speaker,batch_padding_token, batch_padding_attention_mask, audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)
        elif self.args.task == "audio_parsing_probing":
            
            tensor_zeros = torch.zeros((2, 3))  # 오류 방지. 반환용 빼곤 사용되지 않음. 
            inputs =  self.tokenizer(batch_text_input[0], padding=True,truncation=True, return_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return None,None,batch_padding_token, batch_padding_attention_mask, tensor_zeros ,torch.tensor(batch_sr), torch.tensor(batch_vad)

        elif self.args.task == "audio_parsing_with_text_concat":
            batch_audio = [audio.squeeze().flatten() for audio in batch_audio]
            audio_input_values = self.processor(padding=True,audio=batch_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            audio_input_values = torch.tensor(audio_input_values).squeeze(dim=0)
            

            #print(batch_text_input)
            #print("batch_text_input",len(batch_text_input))
 
            inputs =  self.tokenizer(batch_text_input, padding=True,truncation=True, return_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return batch_dialog_id,batch_speaker,batch_padding_token, batch_padding_attention_mask, audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)  
        elif self.args.task == "intonation_contour_with_text":
            audio_input_values = torch.stack(batch_audio)
            #audio_input_values = audio_input_values * 0.00001 #torch.sqrt(torch.abs(audio_input_values))
            
            inputs =  self.tokenizer(batch_text_input, padding=True,truncation=True, return_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return batch_dialog_id,batch_speaker,batch_padding_token, batch_padding_attention_mask, audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)  
        elif self.args.task in [ "intonation_and_wav2vec2", "interpolated_intonation_and_wav2vec2"]:
            # -- pitchs -- #
            audio_input_values = torch.stack(batch_audio)
            audio_input_values = audio_input_values #* 0.000001 #torch.sqrt(torch.abs(audio_input_values))
            
            # -- raw audio -- #
            batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
            tf_audio = self.processor(
                audio=batch_raw_audio,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            ).input_values
            tf_audio = torch.tensor(tf_audio)#.squeeze()
            
            # -- text -- #
            inputs =  self.tokenizer(batch_text_input, padding=True,truncation=True, return_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            target_dtype = torch.long if self.classification else torch.float32
            return batch_dialog_id,batch_speaker,batch_padding_token, batch_padding_attention_mask, (audio_input_values,tf_audio), torch.tensor(batch_sr), torch.tensor(batch_vad, dtype=target_dtype)              

        elif self.args.task in [ "temporal_intonation_and_wav2vec2"]:
            # -- pitchs -- #
            audio_input_values = torch.stack(batch_audio)
            audio_input_values = audio_input_values
            
            # -- raw audio -- #
            batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
            #tf_audio = self.processor(padding=True,audio=batch_raw_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            #tf_audio = self.processor(audio=batch_raw_audio,sampling_rate=16000,return_tensors="pt").input_features # whisper
            #tf_audio = torch.tensor(tf_audio)#.squeeze()

            max_length = 96000
            
            # 1. Truncation: 각 음성 데이터를 max_length에 맞게 자르기
            truncated_audio = [audio[:max_length] if len(audio) > max_length else audio for audio in batch_raw_audio]
            
            # 2. Padding: max_length보다 짧은 경우 0으로 채우기
            padded_audio = [
                torch.cat([torch.tensor(audio), torch.zeros(max_length - len(audio))]) if len(audio) < max_length else torch.tensor(audio)
                for audio in truncated_audio
            ]
            
            # 3. Attention mask 생성: 1은 유효 데이터, 0은 패딩 부분
            attention_masks = [
                torch.cat([torch.ones(audio.shape[0]), torch.zeros(max_length - audio.shape[0])]) if audio.shape[0] < max_length else torch.ones(max_length)
                for audio in truncated_audio
            ]


            # 4. Tensor 변환
            tf_audio = torch.stack(padded_audio)
            attention_masks = torch.stack(attention_masks)
            
            
            return None,None,None, attention_masks, (audio_input_values,tf_audio), torch.tensor(batch_sr), torch.tensor(batch_vad)              

        elif self.args.task == "msp_intonation_and_wav2vec2":
            # -- pitchs -- #
            audio_input_values = torch.stack(batch_audio)
            audio_input_values = audio_input_values #* 0.000001 #torch.sqrt(torch.abs(audio_input_values))
            
            # -- raw audio -- #
            batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
            tf_audio = self.processor(padding=True,audio=batch_raw_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            tf_audio = torch.tensor(tf_audio)#.squeeze()
            
            # -- text -- #
            #inputs =  self.tokenizer(batch_text_input, padding=True,truncation=True, return_tensors="pt") # 
            #batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return None,None,None, None, (audio_input_values,tf_audio), torch.tensor(batch_sr), torch.tensor(batch_vad)              

        elif self.args.task == "mfcc_wav2vec2":
            # -- pitchs -- #
            mfcc_features = torch.stack(batch_audio)
            #audio_input_values = audio_input_values
            mfcc_input = mfcc_features.unsqueeze(0)
            #print("mfcc_input 1 : ", mfcc_input.shape)
            
            # -- raw audio -- #
            batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
            tf_audio = self.processor(padding=True,audio=batch_raw_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            tf_audio = torch.tensor(tf_audio)#.squeeze()
            
            #print("tf_audio.shape : ", tf_audio.shape)
            # -- text -- #
            inputs =  self.tokenizer(batch_text_input, padding=True,truncation=True, retufrn_tensors="pt") # 
            batch_padding_token, batch_padding_attention_mask = inputs['input_ids'], inputs['attention_mask']
            return batch_dialog_id,batch_speaker,batch_padding_token, batch_padding_attention_mask, (mfcc_input,tf_audio), torch.tensor(batch_sr), torch.tensor(batch_vad)              

        elif self.args.task == "double_wav2vec2":
            batch_audio = [audio.squeeze().flatten() for audio in batch_audio]
            audio_input_values = self.processor(padding=True,audio=batch_audio,sampling_rate=16000,return_tensors="pt").input_values #원복해야되는 코드
            #print("audio_input_value : ", audio_input_values.shape)
            audio_input_values = torch.tensor(audio_input_values)#.squeeze()
            return None,None,None,None,audio_input_values,torch.tensor(batch_sr), torch.tensor(batch_vad)        
        
        # -- masked features -- #  
        elif self.args.task == "masked_intonation":
            # -- pitchs -- #
            pitch_values = torch.stack([torch.tensor(pv) for pv in batch_audio])
            
            # Mask some pitch values (e.g., 20% masked)
            mask = torch.rand(pitch_values.shape) < 0.2
            pitch_values[mask] = -1  # Masked values set to -1

            # -- raw audio -- #
            batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
            input_ = self.processor(audio=batch_raw_audio, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            tf_audio = torch.tensor(input_.input_values)
            masked_audio = torch.tensor(input_.attention_mask)

            # Pad pitch_values to match the longest sequence length in the batch
            max_length = max(pv.size(0) for pv in pitch_values)
            pitch_values_padded = torch.full((pitch_values.size(0), max_length), fill_value=-1)
            
            for i, pv in enumerate(pitch_values):
                pitch_values_padded[i, :pv.size(0)] = pv

            # Pad pitch_labels similarly to match the longest sequence length in the batch
            pitch_labels_padded = torch.full((pitch_values_padded.size(0), max_length), fill_value=-1)

            return None, None, tf_audio, masked_audio, pitch_labels_padded, torch.tensor(batch_sr), torch.tensor(batch_vad)

        elif self.args.task == "masked_mfcc":
            # -- pitchs -- #
            mfcc_values = torch.stack([torch.tensor(mv) for mv in batch_audio])
            
            # Mask some pitch values (e.g., 20% masked)
            mask = torch.rand(mfcc_values.shape) < 0.2
            mfcc_values[mask] = -1  # Masked values set to -1

            # -- raw audio -- #
            batch_raw_audio = [audio.squeeze().flatten() for audio in batch_raw_audio]
            input_ = self.processor(audio=batch_raw_audio, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            tf_audio = torch.tensor(input_.input_values)
            masked_audio = torch.tensor(input_.attention_mask)

            #print("mfcc : ", mfcc_values.shape)
            mfcc_values = mfcc_values.squeeze(dim=0).squeeze(dim=-1)
            return None, None, tf_audio, masked_audio, mfcc_values, torch.tensor(batch_sr), torch.tensor(batch_vad)
