#导入文件和目录处理模块os以及数据处理包pandas
import pandas as pd
import os
#定义函数datagena，输入为机组文件夹路径path_machine，输出为整合后csv数据文件的存放路径out_machine
def datagena(path_machine,out_machine):
    #读取机组文件夹下各数据组文件夹的路径并按文件名排序，保证整合时
    #数据组文件夹读取的顺序固定，使接下来的读取以a-e或1-5的顺序进行
    dirs_machine = os.listdir(path_machine)
    dirs_machine.sort()
    rpath_machine = [os.path.join(path_machine,name) for name in dirs_machine]
    #建立输入数据的dataframe对象
    df_machinerow = pd.DataFrame()
    #建立循环，对机组内数据组从a-e或1-5依次进行读取，进行数据整合
    for j in range(0,len(rpath_machine)):
        path_class = rpath_machine[j]
        label = dirs_machine[j]
        print("The name of datagroup is " + label)
        #读取数据组文件夹下各传感器采集文件夹的路径
        #使接下来的读取按照联端X、联端Y、非联端X、非联端Y、轴向A、轴向B进行
        dirs_class = os.listdir(path_class)
        dirs_class.sort()
        rpath_class = [os.path.join(path_class,name) for name in dirs_class]
        #print(rpath_class)
        #这里读取出所有传感器文件夹下包含wave文件的个数，计算出一个最小值，
        #以这个最小值作为读取文件数上限，使读取出的每一条数据都包含六个传感器的采集wave文件中数据
        minfile = []
        for k in range(0,len(rpath_class)):
            DIR = rpath_class[k]
            minfile.append(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
        minnum = min(minfile)
        #print(minnum)
        #建立循环，将本机组该数据组中所有名为wave_i的六个传感器的数据整合到一行
        df_classrow = []
        for i in range(1,minnum+1):
            filename = 'wave_' + str(i) + '.csv'
            print("Now processing " + filename)
            #部分机组可能有rpath_class[6]即轴向C，在这里我们只读取前六个传感器数据
            file1 = os.path.join(rpath_class[0],filename)
            file2 = os.path.join(rpath_class[1],filename)
            file3 = os.path.join(rpath_class[2],filename)
            file4 = os.path.join(rpath_class[3],filename)
            file5 = os.path.join(rpath_class[4],filename)
            file6 = os.path.join(rpath_class[5],filename)
            df_onerow = []
            for filek in [file1,file2,file3,file4,file5,file6]:
                df_wave = pd.read_csv(filek,header=None,skiprows=5)
                df_else = pd.read_csv(filek,header=None,nrows=5)
                df_all = []
                df_all.append(float(df_else.iloc[0,1]))
                df_all.append(float(df_else.iloc[3,1]))
                df_all.extend(df_wave.iloc[0,1:])
                df_onerow.extend(df_all)
                del df_wave,df_else,df_all
            #在每一行最后增加两个属性，说明该行数据来自哪个机组数据组以及来源文件名
            df_onerow.append(label)
            df_onerow.append('wave_' + str(i))
            df_classrow.append(df_onerow)
        df_classrow = pd.DataFrame(df_classrow)
        #每个数据组的数据提取结束后，整合进该机组已提取完毕的数据中
        df_machinerow = pd.concat([df_machinerow,df_classrow])
    df_out = pd.DataFrame(df_machinerow)
    #输出到csv文件
    df_out.to_csv(out_machine,index=False)

#对全部18台机组的数据进行整合
datagena("/dataset/training_data/M1","/datacsv/train01.csv")
datagena("/dataset/training_data/M2","/datacsv/train02.csv")
datagena("/dataset/training_data/M3","/datacsv/train03.csv")
datagena("/dataset/training_data/M4","/datacsv/train04.csv")
datagena("/dataset/training_data/M5","/datacsv/train05.csv")
datagena("/dataset/training_data/M6","/datacsv/train06.csv")
datagena("/dataset/training_data/M7","/datacsv/train07.csv")
datagena("/dataset/training_data/M8","/datacsv/train08.csv")
datagena("/dataset/training_data/M9","/datacsv/train09.csv")
datagena("/dataset/training_data/M10","/datacsv/train10.csv")
datagena("/dataset/testing_data/M11","/datacsv/test11.csv")
datagena("/dataset/testing_data/M12","/datacsv/test12.csv")
datagena("/dataset/testing_data/M13","/datacsv/test13.csv")
datagena("/dataset/testing_data/M14","/datacsv/test14.csv")
datagena("/dataset/testing_data/M15","/datacsv/test15.csv")
datagena("/dataset/testing_data/M16","/datacsv/test16.csv")
datagena("/dataset/testing_data/M17","/datacsv/test17.csv")
datagena("/dataset/testing_data/M18","/datacsv/test18.csv")