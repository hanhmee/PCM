import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC, HubertModel, WhisperModel
from transformers import RobertaModel
from transformers import BertModel
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Define a simple linear classifier
class AudioClassifier(torch.nn.Module):
    def __init__(self,args):
        super(AudioClassifier, self).__init__()
        
        self.args = args

        # -- text -- #
        if self.args.text == "ok":
            self.text_model = RobertaModel.from_pretrained(self.args.text_model_name).cuda()
            self.t_input_dim = self.text_model.config.hidden_size

        # -- audio -- #
        self.audio_model = None
        self.a_hidden_size = 0
        
        if self.args.task in ['normal','msp_normal', 'msp_intonation_and_wav2vec2','audio_parsing', 'double_wav2vec2', 'normal_with_text', 'audio_parsing_with_text','audio_parsing_with_text_concat','intonation_and_wav2vec2','interpolated_intonation_and_wav2vec2','mfcc_wav2vec2']:
            print("task : ", self.args.task)
            self.audio_model = Wav2Vec2Model.from_pretrained(self.args.audio_model_name, mask_feature_length=10).cuda()
            self.a_hidden_size = self.audio_model.config.hidden_size 
        elif self.args.task in ['temporal_normal']:
            print("task : ", self.args.task)
            self.audio_model = Wav2Vec2Model.from_pretrained(self.args.audio_model_name, mask_feature_length=10)
            self.a_hidden_size = self.args.trun_length
        elif self.args.task in ['temporal_intonation_and_wav2vec2']:
            print("task : ", self.args.task)
            self.audio_model = Wav2Vec2Model.from_pretrained(self.args.audio_model_name, mask_feature_length=10)
            self.a_hidden_size = self.args.trun_length
        elif self.args.task in ['ctc_normal']:
            print("task : ", self.args.task)
            self.audio_model = Wav2Vec2ForCTC.from_pretrained(self.args.audio_model_name).cuda()
            self.a_hidden_size = self.audio_model.config.hidden_size
        elif self.args.task in ['contour_image_and_ctc','contour_image_vad']:
            print("task : ", self.args.task)
            self.audio_model = Wav2Vec2ForCTC.from_pretrained(self.args.audio_model_name).cuda()
            self.a_hidden_size = self.audio_model.config.hidden_size
            # -- image -- #
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
            self.linear = nn.Linear(4 * 150 * 750, self.a_hidden_size)
            
        elif self.args.task in ['intonation_contour']:
            print("task : ", self.args.task)
            self.a_hidden_size = 50
        elif self.args.task in ['intonation_contour_with_text']:
            print("task : ", self.args.task)
            self.a_hidden_size = self.t_input_dim # 768 / 1024
        
        # -- linear classifier -- #
        self.num_classes = 4 if getattr(self.args, "objective", "regression") == "classification" else 3
        self.num_mfcc = 13
        
        if self.args.task in ['normal','audio_parsing', 'intonation_contour','ctc_normal','msp_normal','temporal_normal']:
            ##
            self.rnn = nn.LSTM(
            input_size=1024,  # Transformer hidden_dim
            hidden_size=512,  # RNN hidden_dim
            num_layers=1,  # RNN 레이어 개수
            batch_first=True,  # (batch, seq, feature) 형태
            )
            self.a_hidden_size = 512
            ##
            self.linear = torch.nn.Linear(self.a_hidden_size, self.num_classes)
        elif self.args.task in ['temporal_intonation_and_wav2vec2']:
            """
            ##
            self.rnn = nn.LSTM(
            input_size=1024,  # Transformer hidden_dim
            hidden_size=1024,  # RNN hidden_dim
            num_layers=1,  # RNN 레이어 개수
            batch_first=True,  # (batch, seq, feature) 형태
            )
            self.a_hidden_size = 1024
            ##
            # 어텐션 레이어
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
        
            # Pitch embedding
            #self.pitch_embedding = torch.nn.Linear(self.a_hidden_size, self.a_hidden_size)
            # CNN 기반 임베딩 레이어
            self.pitch_embedding = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Conv1D
                nn.ReLU(),  # 활성화 함수
                nn.Conv1d(in_channels=32, out_channels=self.a_hidden_size, kernel_size=3, stride=1, padding=1),  # Conv1D
                nn.AdaptiveAvgPool1d(1)  # 시간축 평균 풀링
            )
            
            self.linear = torch.nn.Linear(self.a_hidden_size, self.num_classes)
            """
            ### fractal self attention 
            self.window_size = 32  # Fractal attention 윈도우 크기
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
            pitch_dim = 1
            self.pitch_embedding = torch.nn.Linear(pitch_dim, self.a_hidden_size)
            self.linear = torch.nn.Linear(self.a_hidden_size, self.num_classes)
            ###
            
        elif self.args.task in ['audio_parsing_probing']:
            print("task : ", self.args.task)
            self.linear = torch.nn.Linear(self.t_input_dim, self.num_classes)
        elif self.args.task in ['normal_with_text','audio_parsing_with_text_concat']:
            self.linear = torch.nn.Linear(self.a_hidden_size + self.t_input_dim, self.num_classes)
            # 어텐션 레이어
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
            self.fc = torch.nn.Linear(self.a_hidden_size, self.num_classes)
        elif self.args.task in ['audio_parsing_with_text']:
            sentence_len = 10
            self.max_len = (self.a_hidden_size + self.t_input_dim) * sentence_len
            self.linear = torch.nn.Linear(self.max_len , self.num_classes).cuda()
        elif self.args.task in ['intonation_and_wav2vec2','interpolated_intonation_and_wav2vec2','intonation_contour_with_text','msp_intonation_and_wav2vec2']:
            if self.args.embedding =='cnn':
                # CNN 기반 임베딩 레이어
                self.pitch_embedding = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Conv1D
                    nn.ReLU(),  # 활성화 함수
                    nn.Conv1d(in_channels=32, out_channels=self.a_hidden_size, kernel_size=3, stride=1, padding=1),  # Conv1D
                    nn.AdaptiveAvgPool1d(1)  # 시간축 평균 풀링
                )
            elif self.args.embedding =='linear':
                self.pitch_embedding = torch.nn.Linear(self.a_hidden_size, self.a_hidden_size) # linear embedding            

            #attention
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
            # Linear layer for emotion prediction
            self.fc = torch.nn.Linear(self.a_hidden_size, self.num_classes)
            
        elif self.args.task in ['mfcc_wav2vec2']:
            # 어텐션 레이어
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
        
            # Pitch embedding
            self.mfcc_embedding = torch.nn.Linear(self.num_mfcc, self.a_hidden_size)
            # Linear layer for emotion prediction
            self.fc = torch.nn.Linear(self.a_hidden_size, self.num_classes)
        elif self.args.task in ['double_wav2vec2']:
            # 어텐션 레이어
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
            # Linear layer for emotion prediction
            self.fc = torch.nn.Linear(self.a_hidden_size, self.num_classes)
        elif self.args.task in ['contour_image_and_ctc','contour_image_vad']:
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=4)
            self.fc = torch.nn.Linear(self.a_hidden_size * 2 , self.num_classes)
 
    def freeze_prameters(self,):
        self.audio_model.feature_extractor._freeze_parameters()
    
    def get_hidden_states(self,):
        return self.hidden_states
    
    def forward(self,pitch_audio, audio_inputs,text_input_tokens, text_attention_mask ):
        
        if self.args.task in ['normal','audio_parsing','msp_normal']:
            
            #print("audio_input : ", audio_inputs.shape)
            #print(audio_inputs)
            if self.args.layer == "last":
                hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).last_hidden_state #hubert, w2v2
                #hidden_states = self.audio_model.encoder(audio_inputs).last_hidden_state # whisper 
            else:
                layer_num = self.args.layer
                hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[layer_num] # hubert, w2v2
                #hidden_states = self.audio_model.encoder(audio_inputs, output_hidden_states=True).hidden_states[layer_num] # whisper
            #print("hidden_states : ", hidden_states.shape)
            #print(hidden_states)
            audio_output = hidden_states.mean(dim=1)
            output = self.linear(audio_output)
        
        if self.args.task in ['temporal_normal']:

            hidden_states = self.audio_model(audio_inputs, attention_mask=text_attention_mask, output_hidden_states=True).last_hidden_state
            ##
            # hidden_dim 기준 풀링: (batch_size, seq_len)
            #pooled_output = hidden_states.mean(dim=-1)  # hidden_dim 방향 평균 풀링
            # 리니어 레이어: (batch_size, seq_len) → 최종 출력
            #output = self.linear(pooled_output)  # output_dim에 따라 차원 축소
            ##
            
            rnn_output, _ = self.rnn(hidden_states)
            final = rnn_output[:, -1, :]  # 마지막 시간 스텝 (batch_size, rnn_hidden_dim)
            output = self.linear(final)

        if self.args.task in ['temporal_intonation_and_wav2vec2']:
            """
            # -- pitchs -- #
            pitches = pitch_audio
             
            # -- raw_audio -- #
            tf_audio = audio_inputs
            wav2vec_outputs = self.audio_model(tf_audio,output_hidden_states=True) #real_w2v2
            #wav2vec_outputs = self.audio_model.encoder(tf_audio,output_hidden_states=True) # whisper
 
            self.hidden_states = wav2vec_outputs.hidden_states
            
            wav2vec_outputs = self.hidden_states[-1]
            # Pitch embedding
            #pitch_embeds = self.pitch_embedding(pitches)
            # CNN 기반 Pitch Embedding
            pitches = pitch_audio.unsqueeze(1)  # (batch_size, 1, seq_len)
            pitch_embeds = self.pitch_embedding(pitches)  # (batch_size, hidden_dim, 1)
            pitch_embeds = pitch_embeds.squeeze(-1)  # (batch_size, hidden_dim)
            
            # Expand pitch embeddings to match wav2vec2.0 output sequence length
            pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)
            
            # Apply attention mechanism
            attention_output, _ = self.attention_layer(query=pitch_embeds, key=self.hidden_states[-1], value=self.hidden_states[-1])#(wav2vec_outputs, pitch_embeds, pitch_embeds)
            
            # 
            rnn_output, _ = self.rnn(attention_output)
            final = rnn_output[:, -1, :]  # 마지막 시간 스텝 (batch_size, rnn_hidden_dim)
            output = self.linear(final)
            """
            ### fractal self attention 
            batch_size = 1
            hidden_dim = self.a_hidden_size
            # -- pitchs -- #
            pitches = pitch_audio
             
            # -- raw_audio -- #
            tf_audio = audio_inputs
            wav2vec_outputs = self.audio_model(tf_audio,output_hidden_states=True) #real_w2v2
            self.hidden_states = wav2vec_outputs.hidden_states
            
            # wav2vec2 피쳐와 피치 임베딩
            wav2vec_outputs = self.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            #print("before  : ", pitches.shape)
            # pitches 텐서를 float32로 변환
            pitches = pitches.to(dtype=torch.float32)
            pitch_embeds = self.pitch_embedding(pitches)  # (batch_size, seq_len, hidden_dim)
            # 시간축 길이를 299로 맞춤
            target_seq_len = 299
            pitch_embeds = self.interpolate_pitch_embeds(pitch_embeds, target_seq_len)
            #print("after : ", pitch_embeds.shape)
            
            # 윈도우 길이 설정
            pitch_seq_len = pitch_embeds.size(1)
            tf_seq_len = wav2vec_outputs.size(1)
            num_windows = min(pitch_seq_len, tf_seq_len) // self.window_size
            wav2vec_windows = wav2vec_outputs[:, :num_windows * self.window_size, :].reshape(batch_size, num_windows, self.window_size, hidden_dim)
            pitch_windows = pitch_embeds[:, :num_windows * self.window_size, :].reshape(batch_size, num_windows, self.window_size, hidden_dim)
            # 윈도우 단위 Attention 계산
            attention_outputs = []
            for i in range(num_windows):
                query = pitch_windows[:, i, :, :]  # 해당 윈도우의 Pitch
                key = wav2vec_windows[:, i, :, :]  # 해당 윈도우의 Key
                value = wav2vec_windows[:, i, :, :]  # 해당 윈도우의 Value

                # 차원 조정 (B, W, H) -> (W, B, H)
                query = query.transpose(0, 1)  # (num_windows, batch_size, hidden_dim)
                key = key.transpose(0, 1)      # (num_windows, batch_size, hidden_dim)
                value = value.transpose(0, 1)  # (num_windows, batch_size, hidden_dim)

                # Cross-Attention 수행
                attention_output, _ = self.attention_layer(query=query, key=key, value=value)
                attention_outputs.append(attention_output)
            
            # 윈도우별 Attention 결과를 결합
            attention_outputs = torch.cat(attention_outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
            
            # attention_outputs 복원
            batch_size = 1  # 실제 배치 크기
            num_windows = attention_outputs.size(0) // batch_size  # 윈도우 수 계산
            hidden_dim = attention_outputs.size(2)  # 은닉 차원

            # reshape하여 batch와 window 분리
            attention_outputs = attention_outputs.view(batch_size, num_windows, -1, hidden_dim)  # (batch_size, num_windows, window_len, hidden_dim)

            # 윈도우 단위의 평균 풀링 수행
            pooled_output = attention_outputs.mean(dim=2)  # (batch_size, num_windows, hidden_dim)

            # 윈도우 전체를 결합한 평균 풀링
            final_output = pooled_output.mean(dim=1)  # (batch_size, hidden_dim)

            # Linear layer에 입력
            output = self.linear(final_output)  # (batch_size, num_classes)
           
        if self.args.task in ['ctc_normal']:
            #print("audio_inputs : ", audio_inputs.shape)
            hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[12]
            #print("hs : ", hidden_states.shape)
            
            audio_output = hidden_states.mean(dim=1)
            vad_output = self.linear(audio_output)
            
            output = (hidden_states, vad_output)
            
        if self.args.task in ['contour_image_and_ctc']:
            # -- image features -- #
            audio_image = self.pool(F.relu(self.conv1(text_attention_mask)))
            audio_image = self.pool(F.relu(self.conv2(audio_image)))
            audio_image = audio_image.view(audio_image.size(0),-1)
            processed_audio_image = self.linear(audio_image)
            
            # -- w2v2 features -- #
            hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[12]
            w2v2_features = hidden_states.mean(dim=1)
            
            attention_output, _ = self.attention_layer(query=processed_audio_image, key=w2v2_features, value=w2v2_features)

            # Average pooling over the sequence length
            pooled_output = attention_output.mean(dim=0)
            
            vad_output = self.fc(pooled_output)
            
            output = (hidden_states, vad_output)
            
        if self.args.task in ['contour_image_vad']:
            # -- image features -- #
            audio_image = self.pool(F.relu(self.conv1(text_attention_mask)))
            audio_image = self.pool(F.relu(self.conv2(audio_image)))
            audio_image = audio_image.view(audio_image.size(0),-1)
            processed_audio_image = self.linear(audio_image)
            
            # -- w2v2 features -- #
            hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[12]
            w2v2_features = hidden_states.mean(dim=1)
            
            attention_output, _ = self.attention_layer(query=processed_audio_image, key=w2v2_features, value=w2v2_features)
            attention_output2, _2 = self.attention_layer(query=w2v2_features, key=processed_audio_image, value=processed_audio_image)

            # Average pooling over the sequence length
            pooled_output1 = attention_output.mean(dim=0)
            pooled_output2 = attention_output2.mean(dim=0)
            
            concatenated_features = torch.cat((pooled_output1, pooled_output2), dim=0)#audio_output
            
            output = self.fc(concatenated_features)
        
        elif self.args.task in ['intonation_contour']:
            audio_output = audio_inputs
            output = self.linear(audio_output)
            
        elif self.args.task in ['normal_with_text']:
            #-- text features --#
            last_token = self.text_model(input_ids=text_input_tokens, attention_mask=text_attention_mask)['last_hidden_state'] 
            #last_token = first_outs[:,-1,:]# (배치 B , 문장 길이 n , 은닉 차원 h) => (B, h)
            #last_token = first_outs.mean(dim=1) 
            
            # -- audio features --#
            if self.args.layer == "last":
                hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).last_hidden_state
            elif self.args.layer == "layer7":
                hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[7]
            #audio_output = hidden_states.mean(dim=1)
            
            last_token = last_token.to(torch.float32)
            hidden_states = hidden_states.to(torch.float32)  
            
            # 텐서의 차원 변환: (batch_size, seq_len, feature_dim) -> (seq_len, batch_size, feature_dim)
            last_token = last_token.permute(1, 0, 2)
            hidden_states = hidden_states.permute(1, 0, 2)

            # Apply attention mechanism
            attention_output, _ = self.attention_layer(query=hidden_states, key=last_token, value=last_token)

            # Average pooling over the sequence length
            pooled_output = attention_output.mean(dim=0)

            #-- concatenation --#
            #concatenated_features = torch.cat((audio_output, last_token), dim=1)#audio_output
        
            output = self.fc(pooled_output)
            
        elif self.args.task in ['audio_parsing_probing']:
            #-- text features --#
            logits = self.text_model(input_ids=text_input_tokens, attention_mask=text_attention_mask)['last_hidden_state']
            last_token = logits.mean(dim=1)
            
            output = self.linear(last_token)

        elif self.args.task in ['audio_parsing_with_text']:
            #-- text features --#
            first_outs = self.text_model(input_ids=text_input_tokens, attention_mask=text_attention_mask)['last_hidden_state'] 
            #last_token = first_outs[:,-1,:]# (배치 B , 문장 길이 n , 은닉 차원 h) => (B, h)
            last_token = first_outs.mean(dim=1) 
            
            # -- audio features --#
            if self.args.layer == "last":
                hidden_states = self.audio_model(audio_inputs).last_hidden_state
            elif self.args.layer == "layer7":
                hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[7]
            audio_output = hidden_states.mean(dim=1)
            
            last_token = last_token.to(torch.float32)
            audio_output = audio_output.to(torch.float32)  

            #print("last token : ",last_token.is_cuda)
            #print("audio_output : ", audio_output.is_cuda)
            #-- concatenation --#
            audio_output = torch.cat((audio_output, last_token), dim=1)#audio_output
            audio_output = audio_output.view(-1)
            
            if audio_output.shape[0] > self.max_len:
                padded_tensor = audio_output[:self.max_len]
            elif audio_output.shape[0] < self.max_len:
                padding_size = self.max_len - audio_output.shape[0]
                padded_tensor = F.pad(audio_output, (0, padding_size), 'constant', 0)
            else :
                padded_tensor = audio_output
            #self.prelinear = torch.nn.Linear(audio_output.shape[0], self.max_len).cuda()
            #padded_tensor = self.prelinear(audio_output)
            output = self.linear(padded_tensor)
        
        elif self.args.task in ['audio_parsing_with_text_concat']:
            #-- text features --#
            
            first_outs = self.text_model(input_ids=text_input_tokens, attention_mask=text_attention_mask)['last_hidden_state'] 
            #last_token = first_outs[:,-1,:]# (배치 B , 문장 길이 n , 은닉 차원 h) => (B, h)
            last_token = first_outs.mean(dim=1) 
            
            # -- audio features --#
            if self.args.layer == "last":
                hidden_states = self.audio_model(audio_inputs).last_hidden_state
            elif self.args.layer == "layer7":
                hidden_states = self.audio_model(audio_inputs,output_hidden_states=True).hidden_states[6]
            audio_output = hidden_states.mean(dim=1)
            
            last_token = last_token.to(torch.float32)
            audio_output = audio_output.to(torch.float32)  

            #-- concatenation --#
            concatenated_features = torch.cat((audio_output, last_token), dim=1)#audio_output

            #self.prelinear = torch.nn.Linear(audio_output.shape[0], self.max_len).cuda()
            #padded_tensor = self.prelinear(audio_output)
            output = self.linear(concatenated_features)
            
        elif self.args.task in ['intonation_contour_with_text']:
            #-- text features --#
            last_token = self.text_model(input_ids=text_input_tokens, attention_mask=text_attention_mask)['last_hidden_state'] 
            #last_token = first_outs[:,-1,:]# (배치 B , 문장 길이 n , 은닉 차원 h) => (B, h)
            # last_token = first_outs.mean(dim=1)
            
            # -- alignment (meaning) -- #
            last_token = last_token.to(torch.float32)
            audio_inputs = audio_inputs.to(torch.float32)  

            # Pitch embedding
            pitch_embeds = self.pitch_embedding(audio_inputs)
        
            # Expand pitch embeddings to match wav2vec2.0 output sequence length
            # pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, last_token.size(1), -1)
            
            last_token = last_token.permute(1, 0, 2)
            # Apply attention mechanism
            attention_output, _ = self.attention_layer(query=pitch_embeds, key=last_token, value=last_token)#(wav2vec_outputs, pitch_embeds, pitch_embeds)
            
            # Average pooling over the sequence length
            pooled_output = attention_output.mean(dim=0)
            
            # Pass through the linear layer
            output = self.fc(pooled_output)
    
        elif self.args.task in ['intonation_and_wav2vec2','msp_intonation_and_wav2vec2','interpolated_intonation_and_wav2vec2']:
            
            # -- pitchs -- #
            pitches = pitch_audio
             
            # -- raw_audio -- #
            tf_audio = audio_inputs
            classification = getattr(self.args, "objective", "regression") == "classification"
            if classification:
                # Keep the pretrained backbone's stochastic pretraining masks/dropout disabled.
                # eval() does not disable gradients, so downstream fine-tuning remains enabled.
                self.audio_model.eval()
            wav2vec_outputs = self.audio_model(tf_audio,output_hidden_states=True) 
            self.hidden_states = wav2vec_outputs.hidden_states
            
            wav2vec_outputs = self.hidden_states[-1]
            if classification and not torch.isfinite(wav2vec_outputs).all():
                non_finite_layers = [
                    index
                    for index, hidden_state in enumerate(self.hidden_states)
                    if not torch.isfinite(hidden_state).all()
                ]
                non_finite_parameters = [
                    name
                    for name, parameter in self.audio_model.named_parameters()
                    if not torch.isfinite(parameter).all()
                ]
                raise FloatingPointError(
                    "Wav2Vec2 produced non-finite hidden states for classification: "
                    f"non_finite_layers={non_finite_layers}; "
                    f"non_finite_parameters={non_finite_parameters[:10]}; "
                    f"audio_shape={tuple(tf_audio.shape)}; "
                    f"audio_dtype={tf_audio.dtype}; "
                    f"audio_min={tf_audio.min().item()}; "
                    f"audio_max={tf_audio.max().item()}."
                )
            # Pitch embedding
            if self.args.embedding =='linear':  
                pitch_embeds = self.pitch_embedding(pitches) # linear embedding 
            elif self.args.embedding =='cnn':
                if self.args.dataset == 'iemocap':
                    pitches = pitch_audio.unsqueeze(1)  # (batch_size, channel = 1, seq_len)
                elif self.args.dataset == 'msp_podcast':
                    pitches = pitches.squeeze(-1).unsqueeze(1)  # (batch_size, channel = 1, seq_len)
                # Ensure pitches is float32
                pitches = pitches.to(dtype=torch.float32)
                #print("after unsqueezing : ",pitches.shape)
                pitch_embeds = self.pitch_embedding(pitches)  # (batch_size, hidden_dim, 1)
            
            if self.args.dataset == 'iemocap':
                pitch_embeds = pitch_embeds.squeeze(-1)  # (batch_size, hidden_dim)
                # Expand pitch embeddings to match wav2vec2.0 output sequence length
                pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)
                
            elif self.args.dataset == 'msp_podcast':
                pitch_embeds = pitch_embeds.transpose(1, 2)  # (batch_size, new_seq_len, hidden_dim)
                # Optionally expand to match wav2vec2.0 output
                pitch_embeds = pitch_embeds.expand(-1, wav2vec_outputs.size(1), -1)
            
            if classification:
                if not torch.isfinite(pitch_embeds).all():
                    raise FloatingPointError(
                        "Pitch projection produced non-finite values for classification."
                    )
                # MultiheadAttention defaults to [sequence, batch, hidden].
                attention_output, _ = self.attention_layer(
                    query=pitch_embeds.transpose(0, 1),
                    key=wav2vec_outputs.transpose(0, 1),
                    value=wav2vec_outputs.transpose(0, 1),
                )
                if not torch.isfinite(attention_output).all():
                    raise FloatingPointError(
                        "Cross-attention produced non-finite values for classification."
                    )
                pooled_output = attention_output.mean(dim=0)
            else:
                attention_output, _ = self.attention_layer(
                    query=pitch_embeds,
                    key=self.hidden_states[-1],
                    value=self.hidden_states[-1],
                )
                pooled_output = attention_output.mean(dim=1)
            
            # Pass through the linear layer
            output = self.fc(pooled_output)
            if classification and not torch.isfinite(output).all():
                raise FloatingPointError(
                    "Classification head produced non-finite logits."
                )

        elif self.args.task in ['mfcc_wav2vec2']:
            
            # -- mfcc -- #
            mfcc = pitch_audio
            #print("mfcc : ", mfcc.shape) 
            # -- raw_audio -- #
            tf_audio = audio_inputs
            #print("tf_audio : 22222", tf_audio.shape)
            wav2vec_outputs = self.audio_model(tf_audio).last_hidden_state
            
            # Pitch embedding
            mfcc_embeds = self.mfcc_embedding(mfcc)
            
            # Expand pitch embeddings to match wav2vec2.0 output sequence length
            # mfcc_embeds = mfcc_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)
            #if mfcc_embeds.size(1) != wav2vec_outputs.size(1):
            #    mfcc_embeds = torch.nn.functional.interpolate(mfcc_embeds.transpose(1, 2), size=wav2vec_outputs.size(1)).transpose(1, 2)
            # Align the dimensions of mfcc_embeds and wav2vec_outputs
            if mfcc_embeds.size(1) != wav2vec_outputs.size(1):
                # Using interpolation to match the dimensions
                target_length = wav2vec_outputs.size(1)
                mfcc_embeds = torch.nn.functional.interpolate(mfcc_embeds.permute(0, 2, 1), size=target_length).permute(0, 2, 1)
            
            # Transpose to match (sequence_length, batch_size, embed_dim)
            #wav2vec_outputs = wav2vec_outputs.permute(1, 0, 2)
            #mfcc_embeds = mfcc_embeds.permute(1, 0, 2)
            
            #print()
            # Apply attention mechanism
            attention_output, _ = self.attention_layer(query=wav2vec_outputs, key=mfcc_embeds, value=wav2vec_outputs)#(wav2vec_outputs, pitch_embeds, pitch_embeds)
            #print("attention_output : ", attention_output.shape)            
            
            # Average pooling over the sequence length
            pooled_output = attention_output.mean(dim=1)
            #print("pooled_output : ", pooled_output.shape)
            
            # Pass through the linear layer
            output = self.fc(pooled_output)
        elif self.args.task in ['double_wav2vec2']:
            
            # -- raw_audio -- #
            wav2vec_outputs = self.audio_model(audio_inputs,output_hidden_states=True)
            
            layer12 = wav2vec_outputs.last_hidden_state
            layer7 = wav2vec_outputs.hidden_states[7]
            
            # Apply attention mechanism
            attention_output, _ = self.attention_layer(query=layer7, key=layer12, value=layer12)#(wav2vec_outputs, pitch_embeds, pitch_embeds)
            
            # Average pooling over the sequence length
            pooled_output = attention_output.mean(dim=1)
            
            # Pass through the linear layer
            output = self.fc(pooled_output)
        return output
    def interpolate_pitch_embeds(self, pitch_embeds, target_seq_len):
        """
        피치 임베딩 데이터를 보간하여 길이를 target_seq_len으로 조정
        Args:
            pitch_embeds: (batch_size, seq_len, hidden_dim) 형태의 텐서
            target_seq_len: 원하는 출력 시퀀스 길이
        Returns:
            interpolated_pitch_embeds: (batch_size, target_seq_len, hidden_dim) 형태의 텐서
        """
        batch_size, seq_len, hidden_dim = pitch_embeds.size()
        interpolated = []

        for i in range(batch_size):
            # 각 샘플의 피치 데이터에 대해 보간 수행
            pitch_sample = pitch_embeds[i].detach().cpu().numpy()  # detach() 추가
            new_time_points = np.linspace(0, seq_len - 1, target_seq_len)  # 목표 길이에 맞는 시간 포인트 생성
            interpolated_sample = np.array([
                np.interp(new_time_points, np.arange(seq_len), pitch_sample[:, j]) for j in range(hidden_dim)
            ]).T  # 보간 후 (target_seq_len, hidden_dim) 형태로 변환
            interpolated.append(interpolated_sample)

        # 리스트를 텐서로 변환
        interpolated_pitch_embeds = torch.tensor(interpolated, dtype=torch.float32, device=pitch_embeds.device)
        return interpolated_pitch_embeds
