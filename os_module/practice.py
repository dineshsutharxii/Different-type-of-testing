import os
import shutil

def remove_folder(folder):
    try:
        os.rmdir(folder)
        print("Removed folder at : ", folder)
    except:
        shutil.rmtree(folder)
        print(f"Removed folder at : {folder} using shutil")

def remove_files(folder):
    dir_items = os.listdir(folder)
    for filename in dir_items:
        if 'file' in filename:
            os.remove(os.path.join(folder, filename))
            print("Removed file : ", os.path.join(folder, filename))

current_dir = os.getcwd()  #current working dir
print(current_dir)
new_folder = os.path.join(current_dir, "new_folder")
if os.path.exists(new_folder):
    remove_folder(new_folder)
print("Files and folders in current directory : ", os.listdir())
os.mkdir(new_folder)
if os.path.exists(new_folder):
    print("Folders created successfully : ", os.listdir())
else:
    print("Folder has not created")

print("Directory after creating the folder : ", os.listdir())

# if os.listdir(new_folder):
#     remove_files(new_folder)

print(f"{new_folder} Directory before creating the file : ", os.listdir(new_folder))

for i in range(3):
    file_name = os.path.join(new_folder, f"file_{i}.txt")
    with open(file_name, "w+") as file:
        pass
print("new_folder directory after creating the files : ", os.listdir(new_folder))


