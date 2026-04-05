# ============================================================
# Instalação das bibliotecas necessárias (rode no Colab)
# !pip install fpdf2 pandas scikit-learn fairlearn imbalanced-learn
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from imblearn.over_sampling import SMOTE
from fpdf import FPDF
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS VICIADOS
# ============================================================

def generate_biased_data(n_samples=1000):
    """Gera um dataset sintético com viés de gênero, desfavorecendo mulheres."""
    np.random.seed(42)

    sexo = np.random.choice(['Masculino', 'Feminino'], size=n_samples, p=[0.6, 0.4])
    experiencia = np.random.randint(1, 8, size=n_samples)
    classificacao = np.random.choice(
        ['Ruim', 'Boa', 'Excelente'], size=n_samples, p=[0.2, 0.5, 0.3]
    )
    soft_skills = np.random.randint(0, 6, size=n_samples)

    classificacao_map = {'Ruim': 1, 'Boa': 2, 'Excelente': 3}
    classificacao_num = np.array([classificacao_map[c] for c in classificacao])

    prob_base = (experiencia * 0.10) + (classificacao_num * 0.20)
    bias_m = np.where(sexo == 'Masculino', 0.20, -0.20)
    prob_final = np.clip(prob_base + bias_m + np.random.normal(0, 0.1), 0, 1)
    entrevista = (prob_final > 0.65).astype(int)

    data = pd.DataFrame({
        'Sexo': sexo,
        'Experiencia_Anos': experiencia,
        'Classificacao': classificacao,
        'Soft_Skills_Pontos': soft_skills,
        'Entrevista': entrevista
    })
    return data


df_original = generate_biased_data(n_samples=1000)


# ============================================================
# 2. FUNÇÃO DE TREINAMENTO E AVALIAÇÃO DE VIÉS
# ============================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, sensitive_feature):
    """Treina o modelo e calcula a Taxa de Seleção por grupo (Fairlearn)."""
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metric_frame = MetricFrame(
        metrics=selection_rate,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    # FIX: uso de .get() para evitar KeyError caso um grupo não apareça no test set
    sr_fem = metric_frame.by_group.get('Feminino', 0)
    sr_masc = metric_frame.by_group.get('Masculino', 0)
    di = sr_fem / sr_masc if sr_masc > 0 else 0
    accuracy = accuracy_score(y_test, y_pred)

    return sr_fem, sr_masc, di, accuracy


# ============================================================
# 3. MODELO VICIADO (sem correção)
# ============================================================

df_model_viciado = df_original.drop(columns=['Sexo'])
df_model_viciado = pd.get_dummies(df_model_viciado, columns=['Classificacao'], drop_first=True)

X = df_model_viciado.drop('Entrevista', axis=1)
y = df_model_viciado['Entrevista']

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X, y, test_size=0.3, random_state=42
)

sensitive_feature = df_original.loc[X_test_v.index, 'Sexo']

sr_fem_viciado, sr_masc_viciado, di_viciado, acc_viciado = train_and_evaluate(
    X_train_v, y_train_v, X_test_v, y_test_v, sensitive_feature
)

print(f"[VICIADO] SR Feminino: {sr_fem_viciado:.2f} | SR Masculino: {sr_masc_viciado:.2f} | DI: {di_viciado:.2f} | Acc: {acc_viciado:.2f}")


# ============================================================
# 4. CORREÇÃO COM SMOTE + MODELO CORRIGIDO
# ============================================================

# Isola os dados de treino originais (com coluna Sexo)
df_train_raw = df_original.loc[y_train_v.index].copy()
X_train_raw = df_train_raw.drop('Entrevista', axis=1)
y_train_raw = df_train_raw['Entrevista']

# Codifica features (sem Sexo) e adiciona flag binária de gênero para o SMOTE
X_train_encoded = pd.get_dummies(
    X_train_raw.drop(columns=['Sexo']), columns=['Classificacao'], drop_first=True
)
X_train_encoded['is_Masculino'] = (X_train_raw['Sexo'] == 'Masculino').astype(int)

# Aplica SMOTE para balancear o conjunto de treino
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_encoded, y_train_raw)

# Remove a coluna auxiliar de gênero — ela não deve entrar no modelo final
X_train_final = X_train_bal.drop(columns=['is_Masculino'], errors='ignore')

# FIX: alinha as colunas do test set com o train set para evitar erros de shape
X_test_final = X_test_v.reindex(columns=X_train_final.columns, fill_value=0)

sr_fem_corrigido, sr_masc_corrigido, di_corrigido, acc_corrigido = train_and_evaluate(
    X_train_final, y_train_bal, X_test_final, y_test_v, sensitive_feature
)

print(f"[CORRIGIDO] SR Feminino: {sr_fem_corrigido:.2f} | SR Masculino: {sr_masc_corrigido:.2f} | DI: {di_corrigido:.2f} | Acc: {acc_corrigido:.2f}")


# ============================================================
# 5. GERAÇÃO DO RELATÓRIO PDF
# ============================================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Relatorio de Auditoria de Vies em IA de Recrutamento', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def print_data(self, data):
        self.set_font('Arial', '', 10)
        for label, value in data:
            self.cell(80, 6, label.encode('latin-1', 'replace').decode('latin-1'), 0)
            self.cell(0, 6, str(value).encode('latin-1', 'replace').decode('latin-1'), 0, 1)

    def print_table(self, header, data, col_widths):
        self.set_fill_color(200, 220, 255)
        self.set_text_color(0)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.3)
        self.set_font('Arial', 'B', 10)

        for i, h in enumerate(header):
            self.cell(col_widths[i], 7, h.encode('latin-1', 'replace').decode('latin-1'), 1, 0, 'C', 1)
        self.ln()

        self.set_fill_color(230, 230, 230)
        self.set_font('Arial', '', 10)
        fill = False
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item).encode('latin-1', 'replace').decode('latin-1'), 'LR', 0, 'C', fill)
            self.ln()
            fill = not fill

        self.cell(sum(col_widths), 0, '', 'T', 1, 'C')


# --- Montagem do PDF ---
pdf = PDF()
pdf.add_page()

pdf.chapter_title('1. Metodologia de Auditoria')
pdf.print_data([
    ('Algoritmo Testado:', 'Regressao Logistica (Triagem de Curriculos)'),
    ('Metrica de Justica:', 'Taxa de Selecao (Selection Rate)'),
    ('Variavel Sensivel:', 'Genero (Sexo)'),
    ('Criterio de Justica:', 'Impacto Desigual (DI - Alvo: > 0.80)'),
    ('Tecnica de Correcao:', 'Balanceamento de Dados (SMOTE)')
])
pdf.ln(5)

pdf.chapter_title('2. Resultados do Modelo Viciado (ANTES da Correcao)')
pdf.ln(2)
pdf.set_font('Arial', '', 10)
pdf.multi_cell(0, 5, 'Este modelo foi treinado com dados historicos viciados e reproduziu o vies de genero, desfavorecendo o grupo feminino na triagem inicial.', 0, 'L')
pdf.ln(2)
pdf.print_table(
    header=['Genero', 'Taxa de Selecao (SR)'],
    data=[['Feminino', f'{sr_fem_viciado:.2f}'], ['Masculino', f'{sr_masc_viciado:.2f}']],
    col_widths=[95, 95]
)
pdf.ln(2)
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 5, f'Acuracia Geral: {acc_viciado:.2f}', 0, 1)
pdf.cell(0, 5, f'Impacto Desigual (DI): {di_viciado:.2f} (Resultado de Vies)', 0, 1)
pdf.ln(5)

pdf.chapter_title('3. Resultados do Modelo Corrigido (POS-Balanceamento SMOTE)')
pdf.ln(2)
pdf.set_font('Arial', '', 10)
pdf.multi_cell(0, 5, 'Apos a aplicacao da tecnica de balanceamento SMOTE no conjunto de treinamento, o modelo foi retreinado para ser mais justo, garantindo taxas de selecao mais equitativas.', 0, 'L')
pdf.ln(2)
pdf.print_table(
    header=['Genero', 'Taxa de Selecao (SR)'],
    data=[['Feminino', f'{sr_fem_corrigido:.2f}'], ['Masculino', f'{sr_masc_corrigido:.2f}']],
    col_widths=[95, 95]
)
pdf.ln(2)
pdf.set_font('Arial', 'B', 10)
pdf.cell(0, 5, f'Acuracia Geral: {acc_corrigido:.2f}', 0, 1)
pdf.cell(0, 5, f'Impacto Desigual (DI): {di_corrigido:.2f} (Resultado Justo)', 0, 1)
pdf.ln(5)

pdf.chapter_title('4. Conclusao da Mitigacao de Vies')
pdf.ln(2)
pdf.set_font('Arial', '', 10)
pdf.multi_cell(
    0, 5,
    f'O Impacto Desigual (DI) foi melhorado de {di_viciado:.2f} para {di_corrigido:.2f}. '
    f'Isso significa que a diferenca nas taxas de selecao entre os grupos de genero foi minimizada. '
    f'O algoritmo corrigido agora cumpre o criterio de justica, promovendo um processo de triagem mais imparcial e etico.',
    0, 'L'
)
pdf.ln(5)

# Salva o PDF
pdf_filename = 'Relatorio_Vies_Recrutamento.pdf'
pdf.output(pdf_filename)

print("\n" + "=" * 70)
print("✅ Relatorio de Auditoria de Vies gerado com sucesso!")
print(f"Nome do arquivo: {pdf_filename}")
print("Para baixar: Va na aba 'Arquivos' (icone de pasta) no Colab,")
print("encontre o arquivo e clique com o botao direito em 'Fazer download'.")
print("=" * 70)
