# AI_DS_NLP
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast

model_path = 'summarization2.pt'
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizerFast.from_pretrained("hyunwoongko/kobart")

def generate_summary(input_text):
    input_ids = tokenizer.encode(input_text, max_length=500, truncation=True, padding='max_length', return_tensors='pt').to(device)
    summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
data = "본 연구는 코로나19 이후 초등학생의 신체활동량과 신체적자기개념, 그리고 긍정심리자본의 관계를 탐색하고자 하였다. 구체적으로 운동참여가 긍정심리자본에 영향력을 행사하는 과정에서 신체적자기개념의 매개효과를 검증하고자 하였다. 이를 위해 초등학교 6학년에 재학 중인 97명의 학생들을 대상으로 주간운동빈도, 주간운동시간, 신체적자기개념, 긍정심리자본을 측정하는 설문조사를 실시하였고, 수집된 자료를 바탕으로 기술통계분석, 상관분석, 구조방정식 모형 검증을 실시하였다. 본 연구를 수행한 결과, 초등학생들의 운동참여수준(주간운동빈도, 주간운동시간) 은 신체적자기개념 및 긍정심리자본의 모든 하위요인들과 정적 상관관계가 나타났다. 또한 신체적자기개념은 운동참여수준과 긍정심리자본의 관계를 부분적으로 매개하는 것으로 나타났다. 본 연구의 결과를 통해 운동에 자주 참여하거나 운동에 참여하는 시간이 많은 학생일수록 자신의 신체에 대해서 긍정적으로 인식하고, 일상생활에서의 희망, 낙관성, 탄력성, 효능감을 가진다는 것을 확인하였으며, 초등학생들이 운동에 참여함으로써 향상된 신체적자기개념이 긍정심리자본의 향상으로 이어질 수 있다는 것을 실증적으로 규명하였다."
generate_summary(data)
