# nlptoolkits

## Name Suffix Info

```...Kits``` means it is a upper folder.

```...Pack``` means it is a pack in upper folder

```...T``` means it is a Tool module.

## Requirement 

See the requirements.txt for more info. 

Venv should be set as python<=3.11

### Pytorch/Torch

Some of the code, like Stanza, need Torch, however, it is always sensible for
use to pre-download the torch, and torch-vision, for The auto-download by pip
will corrupt your virtual-env. So I recommend you download this package before
go to:
https://pytorch.org/get-started/locally/

Remember, Torch will be auto-download when you use requirements.txt, 
so you should config the cuda and torch before download this package, or it will
be automatically run as CPU-version.

### Libre Office in windows

Libre office is a free software and could easily help you to read the doc, if you do not have one then some of code
could
not run properly.

kindly use the guidance below, I use this to config my cmd in windows, Unix-like/BSD sure be easier.
https://darrengoossens.wordpress.com/2020/03/04/convert-between-formats-using-libreoffice-on-the-command-line/

### stanza(Stanford core nlp)
Stanza always cause errors and do not be worry:
IF any resource error: try:
```stanza.download('en')``` or your language
This version do not show so much errors.


### tesseract
If you want to use OCR, then you have to make sure this app is already download and config in python.
nlptoolkit contain a light function, but you could use PyMupdf(Fitz) or other methods.

### poppler

The OCR may also depends on OCR for pdf reading need firstly use pdf2image to convert to 
files like JPEG or PNG(default). More info please refer to https://github.com/Belval/pdf2image

### stanford nlp
TO see the attibute of the stanford core nlp(python client), please refer to:
https://github.com/stanfordnlp/stanza/blob/main/doc/CoreNLP.proto

FURTHER:
https://stanfordnlp.github.io/stanza/client_usage.html

Moreover, You should not shutdown the process in force. Like close the window of terminal or kill the process.
The code it self use ```with CoreNLPClient(...) as client:``` to close the server properly. whatever it runs
properly or not, the server will be closed properly. 
Kill the process in force will cause the server not be closed properly and cause the port be occupied.

If you really want to use a non-killed server, you can set the 
```nlptoolkits.StanzaKits.CoreNLPServerPack.auto_doc_to_sentences_annotator``` 's kwargs ```start_server``` argument from
```stanza.server.StartServer.FORCE_START``` to ```stanza.server.StartServer.TRY_START```, then the server will be
run on non-killed server. But I do not recommend that.

## java
https://zhuanlan.zhihu.com/p/617251948

Java SE Development Kit 8u391 or EARLIER, or running the server will cause unintended error

(The server could not be closed automatically by Python).

Consider your computer, maybe X64 is always better. Or the Corenlp will not run properly.

## docker

If use docker on windows, always download prerelease version for memory issues.

see: https://github.com/docker/for-win/issues/12944

```powershell
wsl --update --pre-release
```

## references

### https://github.com/MS20190155/Measuring-Corporate-Culture-Using-Machine-Learning

1. The code Modified from https://github.com/MS20190155/Measuring-Corporate-Culture-Using-Machine-Learning. However,
   modified for easily use to classify the text to different classes and give them score. For self using. Precisely,
   nlptoolkits/StanzaKits, nlptoolkits/GensimKits/Wrd2vScorerT.py and nlptoolkits/resources/StopWords_Generic.txt
2. the StopWords_Generic.txt's info could be seen in  https://sraf.nd.edu/textual-analysis/resources/#StopWords
3. global_options.py and requirements.txt also modified
   from https://github.com/MS20190155/Measuring-Corporate-Culture-Using-Machine-Learning
4. The reference codes do not have LICENCE now, if any update the reference LICENSE, this repo will follow on. However,
   this code's structure and inner is mostly different from origin code.

5. LexRankT code was copy from https://github.com/crabcamp/lexrank, where they work same, and the
   License is set as 



