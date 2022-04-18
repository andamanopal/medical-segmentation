clc; clear; close all;

tif_list = dir('stent_data/image/*.tif');

valid_mask = zeros(1024,1024,'logical');
[x,y] = meshgrid(-511:512,-511:512);
r = sqrt(x.*x + y.*y);
valid_mask(r<=512) = true;

image_stacks = [];
label_stacks = [];
paths = [];

for i = 1 : length(tif_list)
    image_path = strcat(tif_list(i).folder,'\',tif_list(i).name);
    label_path = replace(image_path,'image','label');
    label_path = replace(label_path,'.tif','_manual1.tif');
    
    nframes = length(imfinfo(image_path));
    
    clear images labels;
    for j = 1 : nframes
        clc; [i j]
        image_temp = rgb2gray(imread(image_path,j));
        image_temp(~valid_mask) = 0;
        images(:,:,j) = image_temp;
        labels(:,:,j) = imread(label_path,j);        
    end   
    
    valid_frames = find(squeeze(sum(labels,[1 2])) ~= 0);
    
    image_stacks = cat(3,image_stacks,images(:,:,valid_frames));
    label_stacks = cat(3,label_stacks,labels(:,:,valid_frames));
    for vf = valid_frames'
        paths = [ paths; replace(image_path,'.tif',sprintf('_%03d.tif',vf)); ];
    end
end

%%

nstacks = size(image_stacks,3);
train_index = randperm(nstacks,round(nstacks*0.75));
train_index = sort(train_index);

mkdir('dataset_des');
mkdir('dataset_des/train/image');
mkdir('dataset_des/train/mask');
mkdir('dataset_des/valid/image');
mkdir('dataset_des/valid/mask');

for i = 1 : nstacks
    clc; i
    if any(train_index == i)
       imwrite(image_stacks(:,:,i),replace(replace(paths(i,:),'image','dataset_des\train\image'),'tif','bmp'));
       imwrite(label_stacks(:,:,i),replace(replace(paths(i,:),'image','dataset_des\train\mask'),'tif','bmp'));
    else
       imwrite(image_stacks(:,:,i),replace(replace(paths(i,:),'image','dataset_des\valid\image'),'tif','bmp'));
       imwrite(label_stacks(:,:,i),replace(replace(paths(i,:),'image','dataset_des\valid\mask'),'tif','bmp'));        
    end
end
