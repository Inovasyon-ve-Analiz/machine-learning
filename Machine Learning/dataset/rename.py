import pathlib
i = 0
for path in pathlib.Path("anti_drone").iterdir():
    if path.is_file():
        old_name = path.stem


        old_extension = path.suffix


        directory = path.parent

        new_name = "anti_image_" + str(i) + old_extension

        path.rename(pathlib.Path(directory, new_name))
        i+=1
