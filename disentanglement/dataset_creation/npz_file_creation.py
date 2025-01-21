from PIL import Image, ImageOps
import os
import numpy as np
import re
from sklearn import preprocessing
import pandas as pd
import os

path_to_files = os.getcwd()
print("imported libraries")
uk_product_data = pd.read_csv("exp_python_image_table.csv")
uk_product_data.head()

array_of_images = []
gray_array_of_images = []
v_make = []
v_makemodel = []
v_color = []
v_firm = []
v_region = []
v_price = []
v_hp = []
v_mpg = []
v_mpd = []
v_hpwt = []
v_space = []
v_wt = []
v_len = []
v_wid = []
v_ht = []
v_wb = []
filenames = []
v_in_uk_blp = []
v_xi_fe = []
v_shares = []
v_wph = []
print("before for loop")
for img in os.listdir(path_to_files):
	if (img.endswith("jpg") & (uk_product_data['Image_name'].eq(img)).any()):
		print(img)
		name = img
		make = uk_product_data.loc[uk_product_data.Image_name == name, 'Automaker']
		makemodel = uk_product_data.loc[uk_product_data.Image_name == name, 'clustering_ids']
		color = uk_product_data.loc[uk_product_data.Image_name == name, 'Segment']
		firm = uk_product_data.loc[uk_product_data.Image_name == name, 'Firm']
		region = uk_product_data.loc[uk_product_data.Image_name == name, 'Region']
		price = uk_product_data.loc[uk_product_data.Image_name == name, 'Price']
		hp = uk_product_data.loc[uk_product_data.Image_name == name, 'hp']
		mpg = uk_product_data.loc[uk_product_data.Image_name == name, 'mpg']
		mpd = uk_product_data.loc[uk_product_data.Image_name == name, 'mpd']
		hpwt = uk_product_data.loc[uk_product_data.Image_name == name, 'hpwt']	
		space = uk_product_data.loc[uk_product_data.Image_name == name, 'space']
		wt = uk_product_data.loc[uk_product_data.Image_name == name, 'wt']
		length = uk_product_data.loc[uk_product_data.Image_name == name, 'len']
		wid = uk_product_data.loc[uk_product_data.Image_name == name, 'wid']
		ht = uk_product_data.loc[uk_product_data.Image_name == name, 'ht']
		wb = uk_product_data.loc[uk_product_data.Image_name == name, 'wb']
		in_uk_blp = uk_product_data.loc[uk_product_data.Image_name == name, 'in_uk_blp'].to_numpy()
		xi_fe = uk_product_data.loc[uk_product_data.Image_name == name, 'xi_fe']
		shares = uk_product_data.loc[uk_product_data.Image_name == name, 'shares']
		wph = uk_product_data.loc[uk_product_data.Image_name == name, 'wph']
		make = int(make)
		makemodel = int(makemodel)
		color = int(color)
		firm = int(firm)
		region = int(region)
		price = float(price)
		hp = float(hp)
		mpg = float(mpg)
		mpd = float(mpd)
		hpwt = float(hpwt)
		space = float(space)
		wt = float(wt)
		length = float(length)
		wid = float(wid)
		ht = float(ht)
		wb = float(wb)
		xi_fe = float(xi_fe)
		shares = float(shares)
		wph = float(wph)
		single_im = Image.open(img)
		single_im = single_im.resize((128,128),Image.ANTIALIAS)
		gray_image = ImageOps.grayscale(single_im)
		single_array = np.array(single_im)
		gray_array = np.array(gray_image)
		if single_array.shape==(128,128,3):
			array_of_images.append(single_array)
			gray_array_of_images.append(gray_array)
			filenames.append(name)
			v_make.append(make)
			v_makemodel.append(makemodel)
			v_color.append(color)
			v_firm.append(firm)
			v_region.append(region)
			v_price.append(price)
			v_hp.append(hp)
			v_mpg.append(mpg)
			v_mpd.append(mpd)
			v_in_uk_blp.append(in_uk_blp)
			v_hpwt.append(hpwt)
			v_space.append(space)
			v_wt.append(wt)
			v_len.append(length)
			v_wid.append(wid)
			v_ht.append(ht)
			v_wb.append(wb)
			v_xi_fe.append(xi_fe)
			v_shares.append(shares)
			v_wph.append(wph)

# print(v_make)
# print(v_makemodel)
# print(v_color)
# print(v_firm)
# print(v_region)
# print(v_price)
# print(v_hp)
# print(v_mpg)
# print(v_mpd)
# print(filenames)

pre = preprocessing.LabelEncoder()

pre.fit(v_make)
post_make=pre.transform(v_make)
make_one_hot_encode=pd.get_dummies(post_make)

pre.fit(v_makemodel)
post_makemodel=pre.transform(v_makemodel)
makemodel_one_hot_encode=pd.get_dummies(post_makemodel)

pre.fit(v_color)
post_color=pre.transform(v_color)
color_one_hot_encode=pd.get_dummies(post_color)

pre.fit(v_firm)
post_firm=pre.transform(v_firm)
firm_one_hot_encode=pd.get_dummies(post_firm)

pre.fit(v_region)
post_region=pre.transform(v_region)
region_one_hot_encode=pd.get_dummies(post_region)

v_make = make_one_hot_encode.to_numpy()
v_makemodel = makemodel_one_hot_encode.to_numpy()
v_color = color_one_hot_encode.to_numpy()
v_firm = firm_one_hot_encode.to_numpy()
v_region = region_one_hot_encode.to_numpy()

print(v_in_uk_blp)
print(v_make.shape)
print(v_makemodel.shape)
print(v_color.shape)
print(v_firm.shape)
print(v_region.shape)

v_make = np.delete(v_make,np.s_[9],axis=1)
v_makemodel = np.delete(v_makemodel,np.s_[0],axis=1)
v_color = np.delete(v_color,np.s_[6],axis=1)
v_firm = np.delete(v_firm,np.s_[16],axis=1)
v_region = np.delete(v_region,np.s_[6],axis=1)

np.savez("cars_original.npz",cars=array_of_images,make=v_make,makemodel=v_makemodel,color=v_color,firm=v_firm,region=v_region,price=v_price,hp=v_hp,mpg=v_mpg,mpd=v_mpd,filenames=filenames,in_uk_blp=v_in_uk_blp,hpwt=v_hpwt,space=v_space,wt=v_wt,length=v_len,wid=v_wid,ht=v_ht,wb=v_wb,xi_fe=v_xi_fe,shares=v_shares,wph=v_wph)
