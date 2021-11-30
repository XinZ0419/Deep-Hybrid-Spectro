from gain_predict import prediction
from gain_rate import gain_freq


# freq_p, freq_b = gain_freq('D://Xin Zhang/Sensor/conbination/NewData/pulse+breath data/pulse+breath only/pulse 81 breath 14.txt')
freq_p, freq_b = gain_freq('D://Xin Zhang/Sensor/conbination/NewData/for_livedemo/H.txt')
print(freq_p, freq_b)

# predict = prediction('D://Xin Zhang/Sensor/conbination/NewData/All/move-shaking/MOVE-SHAKING - Copy (21).txt')
# predict = prediction('D://Xin Zhang/Sensor/conbination/NewData/All/speak-one/SPEAK-ONE - Copy (43).txt')
# predict = prediction('D://Xin Zhang/Sensor/conbination/NewData/All/speak-two/SPEAK-TWO - Copy.txt')
predict, inference_time = prediction('D://Xin Zhang/Sensor/conbination/NewData/for_livedemo/G.txt')
print(predict, inference_time)

# yes -- no -- one -- two --
# shake -- nod -- stretch --
# nod + no -- nod + yes -- shake + no -- shake + yes
