import os

out_ = open("full_results.txt","w")

out_.write("fragment\tsmiles\tid\tsize\n")

for f in os.listdir(os.getcwd()):

    if "output" and ".smi" in f:

        fi = open(f,"r")

        i = 1
        l_list = ""

        for l in fi.readlines():

            out_.write(l)

            i +=1

        fi.close()

out_.close()

