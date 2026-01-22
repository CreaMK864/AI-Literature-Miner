import os
import PyPDF2
import google.generativeai as genai
import pandas as pd
import json
import time
from tqdm import tqdm

# ================= 設定 =================
# 請把下載好的論文 PDF 全部丟進這個資料夾
PDF_FOLDER = './papers/' 
OUTPUT_EXCEL = 'medical_knowledge_base.xlsx'

# 請輸入你的 API KEY
API_KEY = "YOUR_GEMINI_API_KEY" 

# ================= 核心邏輯 =================

def extract_text_from_pdf(pdf_path):
    """讀取 PDF 文字"""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        # 為了節省 Token，我們通常只需要讀前 10 頁 (通常包含摘要、結果、討論)
        # 如果論文很長，可以讀全文，Gemini 1.5 Flash Context Window 很大，夠用的。
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def analyze_paper_with_gemini(text, filename):
    """用 AI 分析論文並提取結構化數據"""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # 這是最強大的部分：我們要求 AI 輸出特定的 JSON 格式
    prompt = f"""
    You are a medical research assistant. Analyze the following academic paper text.
    Extract quantitative data regarding Nailfold Capillaroscopy.
    
    Target Data points:
    1. Normal Capillary Density (loops/mm)
    2. Normal Apical Width/Diameter (um)
    3. Definitions of Abnormalities (e.g., Giant loop size, Ectasia size)
    4. Disease correlations (e.g., "Giant loops are associated with 80% risk of Scleroderma")
    
    Output strictly in JSON format with this structure (return a list of objects):
    [
        {{
            "Category": "Normal Density" or "Dimension" or "Disease Risk" or "Definition",
            "Parameter": "e.g., Mean Density",
            "Value": "e.g., 9",
            "Unit": "loops/mm",
            "Range": "e.g., 7-12",
            "Context": "e.g., Healthy adults",
            "Source_Text": "Quote the sentence from text",
            "Author_Year": "Extract Author and Year from text if possible"
        }}
    ]
    
    If no relevant data is found, return an empty list [].
    
    Paper Filename: {filename}
    Paper Text Content (truncated):
    {text[:50000]} 
    """
    
    try:
        response = model.generate_content(prompt)
        # 清理回應，確保是純 JSON
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        # 補上檔名
        for item in data:
            item['Filename'] = filename
        return data
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

def main():
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"請建立資料夾 '{PDF_FOLDER}' 並把 PDF 放進去！")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("資料夾內沒有 PDF 檔案。")
        return

    all_extracted_data = []

    print(f"🔍 發現 {len(pdf_files)} 篇論文，開始挖掘數據...")
    
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        
        # 1. 讀文字
        text = extract_text_from_pdf(pdf_path)
        if not text: continue
        
        # 2. AI 分析
        extracted_info = analyze_paper_with_gemini(text, pdf_file)
        
        if extracted_info:
            all_extracted_data.extend(extracted_info)
        
        # 避免 API Rate Limit
        time.sleep(2)

    # 3. 存成 Excel
    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data)
        # 調整欄位順序
        cols = ['Category', 'Parameter', 'Value', 'Range', 'Unit', 'Context', 'Disease_Risk', 'Author_Year', 'Filename', 'Source_Text']
        # 確保所有欄位都存在
        for col in cols:
            if col not in df.columns:
                df[col] = ""
                
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n✅ 成功！數據已匯出至: {OUTPUT_EXCEL}")
        print(f"共提取了 {len(all_extracted_data)} 條關鍵數據。")
    else:
        print("❌ 沒有提取到任何數據。")

if __name__ == "__main__":
    main()