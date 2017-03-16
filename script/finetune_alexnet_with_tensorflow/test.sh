
echo `date +"%Y-%m-%d-%H-%m-%S"`
python ./finetune.py -tl fc8 -bs 64 -ne 3 -fp ../../data/filewriter64 -cp ../../data/checkpoint64
echo `date +"%Y-%m-%d-%H-%m-%S"`
python ./finetune.py -tl fc8 -bs 128 -ne 3 -fp ../../data/filewriter128 -cp ../../data/checkpoint128
echo `date +"%Y-%m-%d-%H-%m-%S"`
python ./finetune.py -tl fc8 -bs 256 -ne 3 -fp ../../data/filewriter256 -cp ../../data/checkpoint256
echo `date +"%Y-%m-%d-%H-%m-%S"`
