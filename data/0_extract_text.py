from wiki_dump_reader import Cleaner, iterate
from nltk.tokenize import sent_tokenize, word_tokenize
import re


def pre_token(text):
    # html char replaces
    text = text.replace("&nbsp;", " ")  # non-breaking space
    text = text.replace("&ndash;", "-")
    text = text.replace("&cent;", "¢")
    text = text.replace("&pound;", "£")
    text = text.replace("&sect;", "§")
    text = text.replace("&copy;", "©")
    text = text.replace("&laquo;", "«")
    text = text.replace("&raquo;", "»")
    text = text.replace("&reg;", "®")
    text = text.replace("&deg;", "°")
    text = text.replace("&plusmn;", "±")
    text = text.replace("&para;", "¶")
    text = text.replace("&middot;", "·")
    text = text.replace("&frac12;", "½")
    text = text.replace("&ndash;", "–")  # dash
    text = text.replace("&mdash;", "—")
    text = text.replace("&lsquo;", "‘")
    text = text.replace("&rsquo;", "’")
    text = text.replace("&sbquo;", "‚")
    text = text.replace("&ldquo;", "“")
    text = text.replace("&rdquo;", "”")
    text = text.replace("&bdquo;", "„")
    text = text.replace("&dagger;", "†")
    text = text.replace("&Dagger;", "‡")
    text = text.replace("&bull;", "•")
    text = text.replace("&hellip;", "…")
    text = text.replace("&prime;", "′")
    text = text.replace("&Prime;", "″")
    text = text.replace("&euro;", "€")
    text = text.replace("&trade;", "™")
    text = text.replace("&asymp;", "≈")
    text = text.replace("&ne;", "≠")
    text = text.replace("&le;", "≤")
    text = text.replace("&ge;", "≥")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    
    # special char replaces
    for quotation in ["“", "”"]:
        text = text.replace(quotation, "\"")
    text = text.replace("–", "-").replace("—", "-")  # ndash, mdash
    text = text.replace("’", "'")

    # spaces
    #text = text.replace("-", " - ")
    text = text.replace("/", " / ")
    #text = re.sub(r"'(.)'", r" ' \1 ' ", text)
    text = text.replace("'", " ' ")
    text = text.replace("=", " = ")

    # remove
    text = text.replace("\\", " ")

    return text


def pos_token(text):
    text = text.replace("``", "\"").replace("''", "\"")
    #text = text.replace("e.g .", "e.g.").replace("i.e .", "i.e.").replace("A.E .", "A.E.").replace("P.S .", "P.S.")
    text = re.sub(r"\.(.) \.", r".\1.", text)
    #text = text.replace("s ' ", "s' ")
    text = re.sub(r"^\. ", r"", text)
    text = text.replace(" ( ) ", " ")
    text = text.replace(" ( )", " ")
    text = text.replace("( ) ", " ")

    return text


wiki_extract = open("wiki.txt", "a")

cleaner = Cleaner()
for title, text in iterate("enwiki-20201101-pages-articles-multistream.xml"):
    if len(text) < 2048:
        continue

    title_written = False
    
    text = cleaner.clean_text(text)
    cleaned_text, _ = cleaner.build_links(text)

    prev_lines = {}
    
    for line in cleaned_text.split("\n"):
        if line[-100:] in prev_lines \
           or len(line) < 128 \
           or line.count("\\") > 1 \
           or line.count("{") > 8 \
           or "|" in line \
           or "\"properties\":" in line \
           or line[0:5] == "poly " \
           or bool(re.search("https?:", line)):
            continue
        prev_lines[line[-100:]] = True
        
        if not title_written:
            tokenized_title = word_tokenize( pre_token(title) )
            tokenized_title = ' '.join(tokenized_title)
            tokenized_title = pos_token( tokenized_title )
            print("====", tokenized_title)
            wiki_extract.write("================ " + tokenized_title + "\n")
            title_written = True
        
        line = pre_token(line)
        
        tokenized_line = word_tokenize(line)
        tokenized_line = ' '.join(tokenized_line)

        tokenized_line = pos_token(tokenized_line)
        tokenized_line = tokenized_line.lower()
        
        wiki_extract.write(tokenized_line + "\n")
    
