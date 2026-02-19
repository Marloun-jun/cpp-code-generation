import shutil
import os

def clean_experiment():
    #Очистка перед новым экспериментом
    folders = ['improved_models', 'data']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Удалена папка: {folder}")
    print("Готово для нового эксперимента!")

if __name__ == "__main__":
    clean_experiment()