import glob
import json
import os
import sys

import pdfplumber
from verbal_utils import verb_parse_methods

verb_section_keywords = {
    "ORD": "ORD – Ordförståelse",
    "LÄS": "Svensk läsförståelse – LÄS",
    "MEK": "MEK – Meningskomplettering",
    "ELF": "ELF – Engelsk läsförståelse",
}

def identify_section_pages(pdf_path, section_keywords):
    section_pages = {k: [] for k in section_keywords}
    last_seen_section = None 
    reader = pdfplumber.open(pdf_path)

    # Vi skapar en normaliserad version av dina keywords för jämförelse
    # Vi tar bort alla varianter av bindestreck och mellanslag
    def normalize(s):
        return s.replace("–", "").replace("-", "").replace(" ", "").lower()

    clean_keywords = {k: normalize(v) for k, v in section_keywords.items()}

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        # Normalisera texten på sidan på samma sätt
        clean_text = normalize(text)
        found_section = False

        for section, clean_kw in clean_keywords.items():
            if clean_kw in clean_text:
                section_pages[section].append(page_number)
                last_seen_section = section
                found_section = True
                break

        if not found_section and last_seen_section:
            section_pages[last_seen_section].append(page_number)
            
    return section_pages

if __name__ == "__main__":
    exam_pdfs_path = sys.argv[1]
    pdf_files = sorted(glob.glob(f"{exam_pdfs_path}/*/*.pdf"))
    
    for pdf_path in pdf_files:
        output_path = pdf_path.replace("exam_pdfs", "exams").replace(".pdf", ".json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if "verb" in pdf_path:
            exam_name = os.path.basename(os.path.dirname(pdf_path))
            print(f"\n--- Bearbetar prov: {exam_name} ---")
            
            if os.path.exists(output_path):
                print(f"Laddar befintliga frågor...")
                with open(output_path, "r", encoding="utf-8") as f:
                    try:
                        exam = json.load(f)
                        # Behåll endast frågor som tillhör de verbala typerna
                        # (Filtrerar bort XYZ, KVA, NOG, DTK om de finns i filen)
                        exam = [q for q in exam if q.get("question_type") in verb_section_keywords]
                    except json.JSONDecodeError:
                        print(f"Varning: Kunde inte läsa {output_path}. Startar tom lista.")
                        exam = []
            else:
                print(f"Ingen tidigare data hittad. Startar ny lista.")
                exam = []

            # Identifiera sidor med normalisering för att klara 2022-10-23
            section_pages = identify_section_pages(pdf_path, verb_section_keywords)
            reader = pdfplumber.open(pdf_path)
            
            # 3. Parsa de delar som saknas i listan
            for k, parse_fcn in verb_parse_methods.items():
                if any(q.get("question_type") == k for q in exam):
                    print(f"Skippar {k} (redan laddat)")
                    continue
                
                if section_pages[k]:
                    print(f"Parsar {k}... (Sidor: {[p + 1 for p in section_pages[k]]})")
                    questions = parse_fcn(reader, section_pages[k])
                    if questions:
                        exam.extend(questions)
                else:
                    print(f"Varning: Hittade inga sidor för del {k}")

            # 4. Sortera och spara
            exam.sort(key=lambda x: x.get("question_number", 0))
            
            with open(output_path, "w") as f:
                f.write(json.dumps(exam, ensure_ascii=False, indent=4))
            print(f"Klart! Sparat till {output_path}")
        else:
            continue