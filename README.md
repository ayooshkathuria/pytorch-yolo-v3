Added channel pruning on a pytorch version of <a href="https://github.com/ayooshkathuria/pytorch-yolo-v3"/>YOLOv3</a>
<br>
Use original darknet cfg and weights file as input.
<br>
It will generate pruned_cfg and pruned_weights which could be used with the original <a href="https://github.com/pjreddie/darknet"/>darknet</a>.
<br>
Also compatible with <a href="https://github.com/AlexeyAB/darknet"/>AlexeyAB</a> darknet modification.
<br>
<br>
There is already a pruned model trained on VOC2007+VOC2012 in the output folder.
<br>
Check out the pruned filters in the pruned_channels.txt
