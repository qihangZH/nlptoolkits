"""initialize the project and insure the project can run"""
import global_options
from pathlib import Path
import nlptoolkits

if __name__ == '__main__':
    print("""REFRESHING THE PROJECT FOLDERS,PLEASE WAIT...""")
    # delete the directory if they already exist
    for outputdir in [global_options.PROCESSED_DATA_FOLDER,
                      global_options.MODEL_FOLDER,
                      global_options.OUTPUT_FOLDER

                      ]:
        nlptoolkits.delete_whole_dir(directory=outputdir)

    """root level dir make"""
    Path(global_options.PROCESSED_DATA_FOLDER).mkdir(parents=False, exist_ok=True)
    Path(global_options.MODEL_FOLDER).mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER).mkdir(parents=False, exist_ok=True)

    """model dir make"""
    Path(global_options.MODEL_FOLDER, "phrases").mkdir(parents=False, exist_ok=True)
    Path(global_options.MODEL_FOLDER, "w2v").mkdir(parents=False, exist_ok=True)

    """processed data dir make"""
    Path(global_options.PROCESSED_DATA_FOLDER, "parsed").mkdir(parents=False, exist_ok=True)
    Path(global_options.PROCESSED_DATA_FOLDER, "unigram").mkdir(parents=False, exist_ok=True)
    Path(global_options.PROCESSED_DATA_FOLDER, "bigram").mkdir(parents=False, exist_ok=True)
    Path(global_options.PROCESSED_DATA_FOLDER, "trigram").mkdir(parents=False, exist_ok=True)

    """output result dir make"""
    Path(global_options.OUTPUT_FOLDER, "dict").mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER, "scores").mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER, "scores", "temp").mkdir(parents=False, exist_ok=True)
    Path(global_options.OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=False, exist_ok=True)

    print("""...DONE""")
