# Tele-Prompter

* On-device teleprompter that helps you become more charismatic by providing you with various quotes to say:
* Demo:
https://user-images.githubusercontent.com/279531/208549243-56047321-4ed1-47d6-bd45-ace74621eab8.mp4

* Requires transformers, torch, the general conda install.
* Run by typing `python main.py`. 
* By default, will run using semantic embeddings. If you want to try the highly experimental finetuned model, run `python main.py llm`.
* Right now this will only work on MacOS as it uses on-device speech recognition (using the `hear` library/app).
* It would be trivial to add whisper, feel free to do so and send me a pull request.


