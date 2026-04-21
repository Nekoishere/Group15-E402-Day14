import zipfile
import xml.etree.ElementTree as ET
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def read_docx(path):
    try:
        with zipfile.ZipFile(path) as docx:
            tree = ET.fromstring(docx.read('word/document.xml'))
            # Find all text elements
            texts = []
            for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                p_text = []
                for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    if t.text:
                        p_text.append(t.text)
                if p_text:
                    texts.append(''.join(p_text))
            
            return '\n'.join(texts)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    text = read_docx('Checklist Lab14.docx')
    print("--- DOCX CONTENT ---")
    print(text)
    print("--------------------")
