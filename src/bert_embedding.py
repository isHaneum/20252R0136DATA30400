import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import config
from utils import seed_everything, check_dir

# 
BATCH_SIZE = 64
MAX_LEN = 128


# 평균 풀링 함수
def mean_pooling(model_output, attention_mask):
    #BERT의 모든 토큰 출력을 평균내어 문장 벡터 생성
    token_embeddings = model_output.last_hidden_state#[batch, seq_len, 768]
    #마스크 확장, padding은 제외
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()#
    #분모 0 방지

    #mean vector
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@torch.no_grad()#
def extract_embeddings(file_path, save_name):
    #device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')#토크나이징
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)#모델 로드
    model.eval()
    #평가

    # 1. 텍스트 파일 읽기
    print(f"Reading {file_path}...")#출력
    texts = []
    pids = []
    
    if not os.path.exists(file_path):#없으면 종료
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t', 1)#앞부분 id, 나머지 텍스트
            if len(parts) < 2:
                # fallback to whitespace split if no tab present
                parts = line.strip().split()

                if len(parts) < 2:
                    continue
                pid = parts[0]
                text_content = " ".join(parts[1:])
            else:
                # format: pid \t text
                pid = parts[0]
                text_content = parts[1]
            
            pids.append(pid)#id 저장
            texts.append(text_content)#텍스트 저장

    print(f"Encoding {len(texts)} samples...")

    # 2. batch 처리 및 인코딩
    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        
        # Tokenization
        encoded = tokenizer(#tensor로 변환
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors='pt'
        ).to(device)
        
        # Model Forward
        output = model(**encoded)
        
        # Mean Pooling
        emb = mean_pooling(output, encoded['attention_mask'])
        all_embs.append(emb.cpu()) #계산 결과는 CPU로  

    # 3. 저장
    check_dir(config.EMB_DIR)
    save_path = os.path.join(config.EMB_DIR, save_name)
    # final tensor 생성
    final_tensor = torch.cat(all_embs, dim=0)#
    torch.save({'pids': pids, 'embeddings': final_tensor}, save_path)
    print(f"✅ Saved embeddings to {save_path} (Shape: {final_tensor.shape})") #확인

def main():
    seed_everything(config.SEED)
    
    # training 데이터 임베딩
    extract_embeddings(config.TRAIN_CORPUS_PATH, "train_emb.pt")
    
    # test 데이터
    extract_embeddings(config.TEST_CORPUS_PATH, "test_emb.pt")

if __name__ == "__main__":
    main()