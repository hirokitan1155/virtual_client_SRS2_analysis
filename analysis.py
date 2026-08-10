import os
import json
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

from janome.tokenizer import Tokenizer
from scipy.stats import spearmanr

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

import torch
from lexicalrichness import LexicalRichness
from openai import OpenAI
from statsmodels.stats.multitest import multipletests

# ==========================
# Configuration
# ==========================

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device=device
)

# number of features: 10-20

def calc_corr(name, x):
    # SRS
    rho_srs, p_srs = spearmanr(x, demo_data["SRS2_total"])

    # K6
    rho_k6, p_k6 = spearmanr(x, demo_data["K6_total"])

    results.append({
        "Feature": name,
        "SRS_rho": rho_srs,
        "SRS_p": p_srs,
        "K6_rho": rho_k6,
        "K6_p": p_k6
    })


AGENDA_CACHE_FILE = "agenda_results.json"


if Path(AGENDA_CACHE_FILE).exists():
    with open(AGENDA_CACHE_FILE, "r", encoding="utf-8") as f:
        agenda_cache = json.load(f)
else:
    agenda_cache = {}

# ==========================
# GPT Agenda Extraction
# ==========================

def extract_agenda(patient_text):

    prompt = f"""
あなたは心理面接データの分析者です。
以下の患者発話から、患者が抱えている主な悩み・課題を抽出してください。

0〜1で強度を評価してください。

項目:
- social_relationship: 人間関係・対人関係
- work_school: 仕事・学校
- future_anxiety: 将来への不安
- self_evaluation: 自己評価・自信の低下
- family: 家族関係
- anxiety: 不安全般
- depression: 抑うつ的内容

患者発話:
{patient_text}

JSON形式のみで返してください。
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": "You analyze clinical dialogue."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = json.loads(
        response.choices[0].message.content
    )

    return result



def save_agenda_cache():
    with open(AGENDA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            agenda_cache,
            f,
            ensure_ascii=False,
            indent=2
        )

# ==========================
# NLP Feature Extraction
# ==========================

start = time.perf_counter()

print("Current working directory:", os.getcwd())


#demographic data
demo_data = pd.read_csv("demo_2.csv") # should be removed 2

print(demo_data)
rho, p = spearmanr(demo_data["SRS2_total"],demo_data["K6_total"])
print("Correlation of K6 and SRS:")
print("rho =", rho)
print("p =", p)

plt.scatter(demo_data["SRS2_total"],demo_data["K6_total"], s=50, alpha=0.7)

plt.xlabel("SRS-2 Total")
plt.ylabel("K6 Total")

plt.tight_layout()

plt.savefig("K6_SRS2_scatter.pdf", bbox_inches="tight")
#plt.show()

# Define the directory path and target extension
dir_path = Path("data/diarizations_original/")
extension = "*.txt"  # Change to your target extension (e.g., "*.csv", "*.json")


t = Tokenizer()
words_spe0 = []
words_spe1 = []

stopwords = {
    "こと", "?", "もの", "ところ", "あっ", "なっ","さ","れ","よう", "1","ため","の", "ん", "し", "それ", "これ", "あれ",
     "日", "今日", "中", "方", "時","何","いる","れる","られる","しれる","お話","いただく","できる","お願い","お話し","いただける","とかす","いかが","かなう"
}

normalization = {
    "分かる": "わかる",
}

file_count=0
Jaccard_array=[]
Cosine_array=[]
Bert_array=[]
Bert_turn=[]
Bert_turn_max=[]
Bert_turn_min=[]
Bert_turn_std=[]
Word_count=[]
Word_count_doc=[]
Word_count_ratio=[]
MTLD_patient = []
MTLD_doctor = []
Mean_words_per_turn = []
Agenda_social = []
Agenda_work = []
Agenda_future = []
Agenda_self = []
Agenda_family = []
Agenda_anxiety = []
Agenda_depression = []

opposite_files = {6,8,10,12,18,25,30,31,32,42,44,46,50,52,57,62,107,113,140,144,158,194,210,212,228,245,261} # where the doctor is counted 1

for file_path in sorted(dir_path.glob(extension), key=lambda p: int(p.stem)):
    if file_path.is_file():
        #content = file_path.read_text(encoding="utf-8")
        print(f"--- Content of {Path(file_path).stem} ---")
        print(file_path)
        file_count=file_count+1
        #print(content)
        doctor_embedding = None

        # tokenizer
        ind_words_spe0 = []
        ind_words_spe1 = []
        word_count=0
        word_count_doc=0

        ind_words_patient = []
        ind_words_doctor = []
        # sentence
        Sentence0 = []
        Sentence1 = []
        Patient_sentences = []
        #turn
        bert_turn = []
        patient_turn_count = 0
        doctor_turn_count = 0
        
        file_id = str(file_path.stem)

        if int(file_id) in opposite_files:
            doctor_id = 1
        else:
            doctor_id = 0

        
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if "Speaker 0" in line or "Speaker 1" in line:
                    speaker_id = int(line.split(":")[0].replace("Speaker ", ""))
                    result=line.split(":", 1)[1]
                    result_space = result.replace(" ", "").replace("　", "") # to eliminate spaces
                    
                    #act_words = [token.surface for token in t.tokenize(result_space)]
                    #print(result_space)

                    if speaker_id == 0:
                        Sentence0.append(result_space)
                    else:
                        Sentence1.append(result_space)

                    # for mover BERT and turn
                    if speaker_id == doctor_id:
                        doctor_turn_count += 1
                        doctor_embedding = model.encode(
                            "passage: "+result_space,
                            normalize_embeddings=True
                        )

                    else:
                        patient_turn_count += 1
                        Patient_sentences.append(result_space)

                        patient_embedding = model.encode(
                            "passage: "+result_space,
                            normalize_embeddings=True
                        )

                        if doctor_embedding is not None:
                            distance = cosine_distances(
                                doctor_embedding.reshape(1,-1),
                                patient_embedding.reshape(1,-1)
                            )[0][0]

                            bert_turn.append(distance)
                        

                    for token in t.tokenize(result_space):
                        pos1, pos2 = token.part_of_speech.split(',')[:2]
                        #print(part_of_speech)
                        if pos1 in {"名詞", "動詞", "形容詞"}:
                            #if pos2 not in {"非自立", "代名詞", "数","固有名詞"}:
                            if pos2 not in {"数"}:
                                if token.base_form not in stopwords and token.surface not in stopwords and len(token.base_form) > 2:
                                    base = normalization.get(token.base_form, token.base_form)
                                    if speaker_id == 0:
                                        words_spe0.append(base)
                                        ind_words_spe0.append(base)
                                    else:
                                        words_spe1.append(base)
                                        ind_words_spe1.append(base)

                                    if speaker_id != doctor_id:
                                        word_count += 1
                                        ind_words_patient.append(base)
                                    else:
                                        word_count_doc += 1
                                        ind_words_doctor.append(base)
                                    
                                    
            # Individual Agenda (AI-supported)
            
            counter0 = Counter(ind_words_spe0)
            with open(str(Path(file_path).stem)+"_out_spe0_words.txt", "w", encoding="utf-8") as f:
                for word_0, count_0 in counter0.most_common(100):
                    f.write(f"{word_0},{count_0}\n")
                    #my_set_0.add(word_0)

            counter1 = Counter(ind_words_spe1)
            with open(str(Path(file_path).stem)+"_out_spe1_words.txt", "w", encoding="utf-8") as f:
                for word_1, count_1 in counter1.most_common(100):
                    f.write(f"{word_1},{count_1}\n")
                    #my_set_1.add(word_1)

            # Jaccard
            my_set_0 = set(counter0)
            my_set_1 = set(counter1)
            union_op = my_set_1 | my_set_0
            inter_op = my_set_1 & my_set_0
            Jaccard=len(inter_op)/len(union_op)
            print("Jaccard: ",Jaccard)
            Jaccard_array.append(Jaccard)
            # Cosine
            vocab = sorted(set(counter0) | set(counter1))

            vec0 = np.array([counter0[w] for w in vocab])
            vec1 = np.array([counter1[w] for w in vocab])

            cosine = np.dot(vec0, vec1) / (
                np.linalg.norm(vec0) * np.linalg.norm(vec1)
            )
            print("Cosine: ",cosine)
            Cosine_array.append(cosine)

            # Individual sentence-BERT scores
            embeddings0 = model.encode(
                ["passage: " + s for s in Sentence0],
                normalize_embeddings=True
            )
            mean_embedding0 = np.mean(embeddings0, axis=0)
            mean_embedding0 = mean_embedding0 / np.linalg.norm(mean_embedding0)
            embeddings1 = model.encode(
                ["passage: " + s for s in Sentence1],
                normalize_embeddings=True
            )
            mean_embedding1 = np.mean(embeddings1, axis=0)
            mean_embedding1 = mean_embedding1 / np.linalg.norm(mean_embedding1)

            bert_distance = cosine_distances(
                mean_embedding0.reshape(1, -1),
                mean_embedding1.reshape(1, -1)
            )[0][0]
            print("Bert: ",bert_distance)
            Bert_array.append(bert_distance)

            #Turn-wise sentence-BERT score
            mean_bert_distance  = np.mean(bert_turn)
            Bert_turn.append(mean_bert_distance)
            print("Bert_turn_mean: ",mean_bert_distance)

            #Turn-wise sentence-BERT score (max)
            mean_bert_distance_max  = np.max(bert_turn)
            Bert_turn_max.append(mean_bert_distance_max)
            print("Bert_turn_max: ",mean_bert_distance_max)

            #Turn-wise sentence-BERT score (min)
            mean_bert_distance_min  = np.min(bert_turn)
            Bert_turn_min.append(mean_bert_distance_min)
            print("Bert_turn_min: ",mean_bert_distance_min)

            #Turn-wise sentence-BERT score (variability)
            mean_bert_distance_std  = np.std(bert_turn, ddof=1)
            Bert_turn_std.append(mean_bert_distance_std)
            print("Bert_turn_std: ",mean_bert_distance_std)

            #Word count and turn
            Word_count.append(word_count)
            print("Word count: ",word_count)
            Word_count_doc.append(word_count_doc)
            print("Word count_doc: ",word_count_doc)
            Word_count_ratio.append(word_count / (word_count + word_count_doc))
            print("Word count_ratio: ",word_count / (word_count + word_count_doc))
            if patient_turn_count > 0:
                Mean_words_per_turn.append(word_count / patient_turn_count)
                print("Mean_words_per_turn:", word_count / patient_turn_count)
            else:
                Mean_words_per_turn.append(np.nan)

            #MTLD
            patient_text = " ".join(ind_words_patient)
            doctor_text = " ".join(ind_words_doctor)
            try:
                MTLD_patient.append(LexicalRichness(patient_text).mtld())
            except Exception:
                MTLD_patient.append(np.nan)
            try:
                MTLD_doctor.append(LexicalRichness(doctor_text).mtld())
            except Exception:
                MTLD_doctor.append(np.nan)

            #Agenda
            
            patient_text_agenda = " ".join(Patient_sentences)

            #file_id = str(Path(file_path).stem)

            if file_id in agenda_cache:
                print(f"Loading cached agenda: {file_id}")
                agenda = agenda_cache[file_id]

            else:
                print(f"Calling OpenAI API: {file_id}")

                agenda = extract_agenda(patient_text_agenda)

                agenda_cache[file_id] = agenda

                save_agenda_cache()

            Agenda_social.append(
                agenda["social_relationship"]
            )

            Agenda_work.append(
                agenda["work_school"]
            )

            Agenda_future.append(
                agenda["future_anxiety"]
            )

            Agenda_self.append(
                agenda["self_evaluation"]
            )

            Agenda_family.append(
                agenda["family"]
            )

            Agenda_anxiety.append(
                agenda["anxiety"]
            )

            Agenda_depression.append(
                agenda["depression"]
            )


# ==========================
# Statistical Analysis
# ==========================

results = []

def calc_corr(name, x):
    # SRS-2
    rho_srs, p_srs = spearmanr(
        x,
        demo_data["SRS2_total"]
    )

    # K6
    rho_k6, p_k6 = spearmanr(
        x,
        demo_data["K6_total"]
    )

    return {
        "feature": name,
        "rho_srs": rho_srs,
        "p_srs": p_srs,
        "rho_k6": rho_k6,
        "p_k6": p_k6
    }


results.append(calc_corr("Jaccard", Jaccard_array))
results.append(calc_corr("Cosine", Cosine_array))
results.append(calc_corr("Bert", Bert_array))
results.append(calc_corr("Bert_turn", Bert_turn))
results.append(calc_corr("Bert_turn_max", Bert_turn_max))
results.append(calc_corr("Bert_turn_min", Bert_turn_min))
results.append(calc_corr("Bert_turn_std", Bert_turn_std))
results.append(calc_corr("Word_count", Word_count))
results.append(calc_corr("Word_count_doc", Word_count_doc))
results.append(calc_corr("Word_count_ratio", Word_count_ratio))
results.append(calc_corr("MTLD_patient", MTLD_patient))
results.append(calc_corr("MTLD_doctor", MTLD_doctor))
results.append(calc_corr("Mean_words_per_turn", Mean_words_per_turn))
results.append(calc_corr("Agenda_social", Agenda_social))
results.append(calc_corr("Agenda_work", Agenda_work))
results.append(calc_corr("Agenda_future", Agenda_future))
results.append(calc_corr("Agenda_self", Agenda_self))
results.append(calc_corr("Agenda_family", Agenda_family))
results.append(calc_corr("Agenda_anxiety", Agenda_anxiety))
results.append(calc_corr("Agenda_depression", Agenda_depression))


results_df = pd.DataFrame(results)


# ==========================
# FDR correction
# ==========================

# Benjamini-Hochberg FDR
# SRS-2 and K6 are corrected separately

reject_srs, p_fdr_srs, _, _ = multipletests(
    results_df["p_srs"],
    alpha=0.05,
    method="fdr_bh"
)

results_df["p_fdr_srs"] = p_fdr_srs
results_df["significant_fdr_srs"] = reject_srs


reject_k6, p_fdr_k6, _, _ = multipletests(
    results_df["p_k6"],
    alpha=0.05,
    method="fdr_bh"
)

results_df["p_fdr_k6"] = p_fdr_k6
results_df["significant_fdr_k6"] = reject_k6


# ==========================
# Print results
# ==========================

print("\n===== SRS-2: FDR corrected =====")

print(
    results_df[
        [
            "feature",
            "rho_srs",
            "p_srs",
            "p_fdr_srs",
            "significant_fdr_srs"
        ]
    ].sort_values("p_srs")
)


print("\n===== K6: FDR corrected =====")

print(
    results_df[
        [
            "feature",
            "rho_k6",
            "p_k6",
            "p_fdr_k6",
            "significant_fdr_k6"
        ]
    ].sort_values("p_k6")
)


# Save
results_df.to_csv(
    "correlation_results_FDR.csv",
    index=False,
    encoding="utf-8-sig"
)


X = pd.DataFrame({
    "Jaccard": Jaccard_array,
    "Cosine": Cosine_array,
    "Bert": Bert_array,
    "Bert_turn": Bert_turn,
    "Bert_turn_max": Bert_turn_max,
    "Bert_turn_min": Bert_turn_min,
    "Bert_turn_std": Bert_turn_std,
    "Word_count": Word_count,
    "Word_count_doc": Word_count_doc,
    "Word_ratio": Word_count_ratio,
    "MTLD_patient": MTLD_patient,
    "MTLD_doctor": MTLD_doctor,
    "Mean_words_per_turn": Mean_words_per_turn,
    "Agenda_social": Agenda_social,
    "Agenda_work": Agenda_work,
    "Agenda_future": Agenda_future,
    "Agenda_self": Agenda_self,
    "Agenda_family": Agenda_family,
    "Agenda_anxiety": Agenda_anxiety,
    "Agenda_depression": Agenda_depression
})


X.to_csv(
    "nlp_features.csv",
    index=False,
    encoding="utf-8-sig"
)

demo_data[
    ["ID", "SRS2_total", "K6_total"]
].to_csv(
    "outcomes.csv",
    index=False,
    encoding="utf-8-sig"
)
