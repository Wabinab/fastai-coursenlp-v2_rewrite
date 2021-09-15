import os
import re
import shutil
from tqdm.auto import tqdm

from fastai.data.external import download_url
from fastcore.foundation import working_directory
from fastcore.xtras import bunzip

def get_wiki(path, lang):
    name = f'{lang}wiki'
    if (path/name).exists(): 
        print(f"{path/name} already exists; not downloading")
        return

    xml_fn = f"{lang}wiki-latest-pages-articles.xml"
    zip_fn = f"{xml_fn}.bz2"

    if not (path/xml_fn).exists():
        print("Downloading...")
        download_url(f"https://dumps.wikimedia.org/{name}/latest/{zip_fn}", path/zip_fn)
        print("Unzipping...")
        bunzip(path/zip_fn)

    with working_directory(path):
        if not (path/"wikiextractor").exists(): os.system("git clone https://github.com/attardi/wikiextractor.git")
        print("Extracting...")
        os.system(f"python -m wikiextractor.wikiextractor.WikiExtractor --processes {os.cpu_count()} --no-templates " +
                f"-b 100G -q {xml_fn}")
    shutil.move(str(path/"text/AA/wiki_00"), str(path/name))
    shutil.rmtree(path/"text")


def split_wiki(path, lang, word_threshold=1800, spaced_lang=True):
    """
    :word_threshold: What threshold to use. If less than this, will delete file. 
    :spaced_lang: Language using space as separator? Defaults: True. 
        If Chinese/Japanese/etc for example, set this to False. 

    Problem that might need fixing: 
        "\n" is counted as 1 instead of 0. 
    """
    dest = path/"docs"
    name = f"{lang}wiki"
    if dest.exists():
        print(f"{dest} already exists; not splitting.")
        return dest

    dest.mkdir(exist_ok=True, parents=True)
    title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
    lines = (path/name).open()
    f = None
    num_words = 0
    num_errors = 0

    for i, l in enumerate(tqdm(lines)):
        # if i % 100000 == 0: print(i)
        if l.startswith('<doc id="'):
            if f: f.close()
            if num_words < word_threshold and i != 0: 
                try: os.remove(str(dest/f"{title}.txt"))
                except Exception: pass
            num_words = 0

            title = title_re.findall(l)[0].replace("/", "_")
            if len(title) > 150: continue
            f = (dest/f"{title}.txt").open("w")
        else: 
            if spaced_lang: num_words += len(l.split(" "))
            else: num_words += len(l)
            try: f.write(l)
            except ValueError: 
                num_errors += 1
                if len(title) > 150: continue
                f = (dest/f"{title}.txt").open("w")
                f.write(l)
    f.close()
    print("Number of errors encountered when writing file: ", num_errors)
    print("Note all errors are resolved internally. ")
    return dest