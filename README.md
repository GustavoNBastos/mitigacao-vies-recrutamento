# 🤖 Mitigação de Viés em IA de Recrutamento

Projeto acadêmico desenvolvido no curso **Descobrindo a IA — 2025**.

Autor: **Gustavo do Nascimento Bastos**

---

## 📋 Sobre o Projeto

Este projeto demonstra como **viés algorítmico** pode surgir em sistemas de Inteligência Artificial utilizados em processos de recrutamento e como ele pode ser identificado, medido e mitigado.

O pipeline simula um sistema de triagem automatizada de currículos e aplica técnicas de auditoria de fairness para avaliar impactos discriminatórios entre grupos.

---

## 🎯 Objetivos

- Gerar dataset sintético com viés de gênero intencional
- Treinar modelo de classificação com viés embutido
- Medir fairness utilizando Fairlearn
- Aplicar SMOTE para mitigação do viés
- Comparar resultados antes e depois da correção

---

## 📊 Resultados

| Métrica | Modelo com Viés | Modelo Corrigido |
|---|---|---|
| Taxa de Seleção — Feminino | 0.29 | 0.53 |
| Taxa de Seleção — Masculino | 0.58 | 0.56 |
| Impacto Desigual (DI) | ≈ 0.50 ❌ | ≈ 0.95 ✅ |

Critério de justiça adotado: **DI ≥ 0.80**

---

## 🏗️ Estrutura do Projeto
mitigacao-vies-recrutamento/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│ └── algoritmo_vies_recrutamento.ipynb
│
└── docs/
└── wiki.md


---

## ⚙️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Fairlearn
- Imbalanced-learn (SMOTE)
- FPDF2

---

## 🚀 Como Executar

### Google Colab (recomendado)

1. Abra o Google Colab
2. Faça upload do notebook
3. Execute:

```python
!pip install fpdf2 pandas scikit-learn fairlearn imbalanced-learn

4.Execute todas as células.
🔬 Metodologia

Pipeline aplicado:

Geração de Dados → Treinamento → Auditoria → Mitigação → Retreinamento → Avaliação

O modelo aprende viés indireto presente nos dados históricos e posteriormente é corrigido via balanceamento utilizando SMOTE.

📖 Conceitos-Chave
Viés Algorítmico
Fairness em Machine Learning
Disparate Impact (DI)
Selection Rate
Balanceamento de Dados
👤 Autor

Gustavo do Nascimento Bastos

Projeto desenvolvido para fins educacionais e de portfólio em Data & AI.

📄 Licença

Distribuído sob licença MIT.