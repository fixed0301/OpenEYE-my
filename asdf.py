import os

def get_files_count(path):
    a = os.listdir(path)
    return len(a)
if __name__ == '__main__':
    print(get_files_count('opened_eyes_dataset'))