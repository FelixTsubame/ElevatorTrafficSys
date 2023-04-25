from random import choices
from random import randint
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# All arrival time are randomized

def workday_rm_generator():
    #hour7
    rm = []

    floor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights_fsm = [0.7465674110835401, 
               0.027460711331679075,
               0.0304383788254756,
               0.027460711331679075,
               0.02663358147229115,
               0.02630272952853598,
               0.030934656741108353,
               0.02663358147229115,
               0.028949545078577336,
               0.028618693134822168]

    weights_strm = [0.15913978494623657, 
               0.09975186104218363,
               0.08651778329197685,
               0.09478908188585608,
               0.09528535980148883,
               0.09826302729528535,
               0.08899917287014061,
               0.09263854425144748,
               0.095616211745244,
               0.08899917287014061]

    hour7_arrivaltime = []
    fsm = []
    strm = []
    for i in range(60):
    	hour7_arrivaltime.append(randint(0, 3600))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour7_arrivaltime.sort()
    # print("hour7")
    # print()
    # print(hour7_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(60):
    	rm.append([hour7_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)
    #hour8

    weights_fsm = [0.7488234751314939, 
               0.028144320383870075,
               0.028697979145519976,
               0.028697979145519976,
               0.025745132416720495,
               0.02731383224139522,
               0.028328873304420042,
               0.02491464427424564,
               0.029805296668819783,
               0.029528467287994832]

    weights_strm = [0.16360616406754636, 
               0.09319922487773369,
               0.08655531973793486,
               0.09172280151333395,
               0.09808987727230783,
               0.10131955338193227,
               0.09513703054350835,
               0.09061548399003415,
               0.08655531973793486,
               0.09319922487773369]

    hour8_arrivaltime = []
    fsm = []
    strm = []
    for i in range(110):
    	hour8_arrivaltime.append(randint(3600, 7200))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour8_arrivaltime.sort()
    # print("hour8")
    # print()
    # print(hour8_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(110):
    	rm.append([hour8_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour9

    weights_fsm = [0.5956050228310502, 
               0.04580479452054795,
               0.04366438356164384,
               0.04052511415525114,
               0.046090182648401826,
               0.046232876712328765,
               0.045234018264840185,
               0.04580479452054795,
               0.04509132420091324,
               0.04594748858447489]

    weights_strm = [0.3117865296803653, 
               0.08304794520547945,
               0.07705479452054795,
               0.0684931506849315,
               0.07876712328767123,
               0.0783390410958904,
               0.07976598173515982,
               0.07106164383561644,
               0.07676940639269407,
               0.07491438356164383]

    hour9_arrivaltime = []
    for i in range(70):
    	hour9_arrivaltime.append(randint(7200, 10800))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour9_arrivaltime.sort()
    # print("hour9")
    # print()
    # print(hour9_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(70):
    	rm.append([hour9_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour10

    weights_fsm = [0.45642201834862384, 
               0.06684141546526867,
               0.05701179554390563,
               0.0563564875491481,
               0.05897771952817824,
               0.06389252948885976,
               0.05930537352555701,
               0.07044560943643513,
               0.05602883355176933,
               0.05471821756225426]

    weights_strm = [0.4079292267365662, 
               0.07601572739187418,
               0.056684141546526866,
               0.06389252948885976,
               0.06520314547837483,
               0.07142857142857142,
               0.06684141546526867,
               0.06422018348623854,
               0.06618610747051114,
               0.061598951507208385]

    hour10_arrivaltime = []
    for i in range(30):
    	hour10_arrivaltime.append(randint(10800, 14400))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour10_arrivaltime.sort()
    # print("hour10")
    # print()
    # print(hour10_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(30):
    	rm.append([hour10_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour11

    weights_fsm = [0.40285996055226825, 
               0.07396449704142012,
               0.07593688362919132,
               0.0601577909270217,
               0.06607495069033531,
               0.0611439842209073,
               0.0670611439842209,
               0.0675542406311637,
               0.06854043392504931,
               0.05670611439842209]

    weights_strm = [0.46696252465483234, 
               0.059664694280078895,
               0.05917159763313609,
               0.05276134122287968,
               0.0641025641025641,
               0.0641025641025641,
               0.05621301775147929,
               0.059664694280078895,
               0.05276134122287968,
               0.0645956607495069]

    hour11_arrivaltime = []
    for i in range(20):
    	hour11_arrivaltime.append(randint(14400, 18000))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour11_arrivaltime.sort()
    # print("hour11")
    # print()
    # print(hour11_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(20):
    	rm.append([hour11_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour12

    weights_fsm = [0.39487132688446797, 
               0.06378901259353896,
               0.0656141631684614,
               0.06634422339843037,
               0.07400985581310458,
               0.06479284540974631,
               0.0677130863296222,
               0.06588793575469977,
               0.06743931374338383,
               0.06953823690454462]

    weights_strm = [0.4694287278700493, 
               0.062420149662347144,
               0.0643365577660157,
               0.04937032305165176,
               0.055302062420149664,
               0.06251140719109327,
               0.0609600292024092,
               0.05429822960394232,
               0.06114254425990144,
               0.06022996897244023]

    hour12_arrivaltime = []
    for i in range(110):
    	hour12_arrivaltime.append(randint(18000, 21600))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour12_arrivaltime.sort()
    # print("hour12")
    # print()
    # print(hour12_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(110):
    	rm.append([hour12_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour13

    weights_fsm = [0.4947578632051922, 
               0.05983881320875829,
               0.05341987019470794,
               0.05477498038656301,
               0.05591612581128307,
               0.054703658797518005,
               0.05691462805791313,
               0.05991013479780329,
               0.05798445189358819,
               0.05177947364667285]

    weights_strm = [0.41566222095428285, 
               0.07018044362028386,
               0.06411810855145852,
               0.07025176520932887,
               0.06240639041437843,
               0.07075101633264388,
               0.05826973824976821,
               0.0656158619214036,
               0.06026674274302832,
               0.06247771200342343]

    hour13_arrivaltime = []
    for i in range(140):
    	hour13_arrivaltime.append(randint(21600, 25200))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour13_arrivaltime.sort()
    # print("hour13")
    # print()
    # print(hour13_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(140):
    	rm.append([hour13_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour14

    weights_fsm = [0.43910925539318024, 
               0.06541405706332637,
               0.06263048016701461,
               0.06889352818371608,
               0.06367432150313153,
               0.05706332637439109,
               0.06819763395963814,
               0.05775922059846903,
               0.06297842727905359,
               0.054279749478079335]

    weights_strm = [0.3747390396659708, 
               0.06784968684759916,
               0.0685455810716771,
               0.07237299930410578,
               0.0720250521920668,
               0.07028531663187196,
               0.06784968684759916,
               0.07063326374391092,
               0.06332637439109255,
               0.07237299930410578]

    hour14_arrivaltime = []
    for i in range(30):
    	hour14_arrivaltime.append(randint(25200, 28800))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour14_arrivaltime.sort()
    # print("hour14")
    # print()
    # print(hour14_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(30):
    	rm.append([hour14_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour15

    weights_fsm = [0.35330578512396693, 
               0.06973140495867769,
               0.07747933884297521,
               0.07179752066115702,
               0.07283057851239669,
               0.07179752066115702,
               0.06456611570247933,
               0.06973140495867769,
               0.07283057851239669,
               0.0759297520661157]

    weights_strm = [0.4731404958677686, 
               0.06766528925619834,
               0.05165289256198347,
               0.054235537190082644,
               0.04855371900826446,
               0.061466942148760334,
               0.05785123966942149,
               0.0625,
               0.06043388429752066,
               0.0625]

    hour15_arrivaltime = []
    for i in range(20):
    	hour15_arrivaltime.append(randint(28800, 32400))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour15_arrivaltime.sort()
    # print("hour15")
    # print()
    # print(hour15_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(20):
    	rm.append([hour15_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)

    #hour16

    weights_fsm = [0.35129237999328633, 
               0.07166834508224236,
               0.07250755287009064,
               0.07972473984558577,
               0.07183618663981202,
               0.07099697885196375,
               0.07200402819738168,
               0.07385028533064787,
               0.06982208794897617,
               0.06629741524001342]

    weights_strm = [0.4696206780798926, 
               0.06377979187646861,
               0.05924806982208795,
               0.054380664652567974,
               0.056059080228264516,
               0.06042296072507553,
               0.05673044645854314,
               0.0572339711312521,
               0.0652903658945955,
               0.0572339711312521]

    hour16_arrivaltime = []
    for i in range(60):
    	hour16_arrivaltime.append(randint(32400, 36000))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour16_arrivaltime.sort()
    # print("hour16")
    # print()
    # print(hour16_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(60):
    	rm.append([hour16_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)


    #hour17

    weights_fsm = [0.14907535539061517, 
               0.09259026292615423,
               0.09762234243301045,
               0.09196125298779721,
               0.0992577682727387,
               0.0988803623097245,
               0.09321927286451126,
               0.09296766888916845,
               0.09070323311108315,
               0.09372248081519688]

    weights_strm = [0.7553151339791169, 
               0.03308592275757957,
               0.026166813435652282,
               0.024279783620581204,
               0.024405585608252612,
               0.03283431878223676,
               0.024279783620581204,
               0.022644357780852938,
               0.03195370486853692,
               0.025034595546609636]

    hour17_arrivaltime = []
    for i in range(80):
    	hour17_arrivaltime.append(randint(36000, 39600))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour17_arrivaltime.sort()
    # print("hour17")
    # print()
    # print(hour17_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(80):
    	rm.append([hour17_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)


    #hour18

    weights_fsm = [0.0975, 
               0.10333333333333333,
               0.10366666666666667,
               0.0925,
               0.09783333333333333,
               0.106,
               0.096,
               0.099,
               0.10366666666666667,
               0.1005]

    weights_strm = [0.7676666666666667, 
               0.027666666666666666,
               0.025666666666666667,
               0.02666666666666667,
               0.026333333333333334,
               0.026833333333333334,
               0.017666666666666667,
               0.025333333333333333,
               0.0265,
               0.029666666666666668]

    hour18_arrivaltime = []
    for i in range(60):
    	hour18_arrivaltime.append(randint(39600, 43200))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour18_arrivaltime.sort()
    # print("hour18")
    # print()
    # print(hour18_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(60):
    	rm.append([hour18_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)


    #hour19

    weights_fsm = [0.10196209961428811, 
               0.09508636592319303,
               0.10464531276203254,
               0.09877578400134161,
               0.09827268153613952,
               0.09458326345799094,
               0.09894348482307563,
               0.10179439879255409,
               0.10498071440550058,
               0.10095589468388395]

    weights_strm = [0.8063055508971994, 
               0.021801106825423446,
               0.01861479121247694,
               0.028844541338252556,
               0.020962602716753313,
               0.021298004360221365,
               0.017943987925540836,
               0.02733523394264632,
               0.01626697970820057,
               0.020627201073285258]

    hour19_arrivaltime = []
    for i in range(60):
    	hour19_arrivaltime.append(randint(43200, 46800))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour19_arrivaltime.sort()
    # print("hour19")
    # print()
    # print(hour19_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(60):
    	rm.append([hour19_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)


    #hour20

    weights_fsm = [0.10440376051459674, 
               0.11281543790202869,
               0.10687778327560614,
               0.09450766947055914,
               0.09203364670954972,
               0.10341415141019297,
               0.10044532409698169,
               0.09698169223156854,
               0.09450766947055914,
               0.09401286491835725]

    weights_strm = [0.801583374567046, 
               0.02127659574468085,
               0.02375061850569025,
               0.027709054923305294,
               0.02078179119247897,
               0.01830776843146957,
               0.02028698664027709,
               0.033151904997525974,
               0.01682335477486393,
               0.01632855022266205]

    hour20_arrivaltime = []
    for i in range(20):
    	hour20_arrivaltime.append(randint(46800, 50400))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour20_arrivaltime.sort()
    # print("hour20")
    # print()
    # print(hour20_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(20):
    	rm.append([hour20_arrivaltime[i], fsm[i], strm[i]])

    # print("rm = ", rm)


    #hour21

    weights_fsm = [0.11346153846153846, 
               0.07692307692307693,
               0.10961538461538461,
               0.1,
               0.11153846153846154,
               0.10384615384615385,
               0.11538461538461539,
               0.08461538461538462,
               0.09807692307692308,
               0.08653846153846154]

    weights_strm = [0.7807692307692308, 
               0.021153846153846155,
               0.021153846153846155,
               0.023076923076923078,
               0.038461538461538464,
               0.019230769230769232,
               0.019230769230769232,
               0.03461538461538462,
               0.025,
               0.01730769230769231]

    hour21_arrivaltime = []
    for i in range(5):
    	hour21_arrivaltime.append(randint(50400, 54000))
    	temp_fsm = choices(floor, weights_fsm)
    	temp_strm = choices(floor, weights_strm)
    	while temp_fsm == temp_strm:
    		temp_fsm = choices(floor, weights_fsm)
    		temp_strm = choices(floor, weights_strm)
    	fsm.append(temp_fsm[0])
    	strm.append(temp_strm[0])

    hour21_arrivaltime.sort()
    # print("hour21")
    # print()
    # print(hour21_arrivaltime)
    # print(fsm)
    # print(strm)
    # print()
    for i in range(5):
    	rm.append([hour21_arrivaltime[i], fsm[i], strm[i]])
     
    # print("rm = ", rm)

    request = []
    with open('pattern_dist.txt', 'w') as f:
        for item in rm:
          temp_r = []
          f.write("%s " % str(item[1]-1))
          temp_r.append(item[1]-1)
          if item[1] - item[2] < 0:
            f.write("1 ")
            temp_r.append(0)
          else:
            f.write("-1 ")
            temp_r.append(1)
          f.write("%s " % str(item[2]-1))
          temp_r.append([item[2]-1])
          f.write("%s\n" % item[0])
          temp_r.append(item[0])
          request.append(temp_r)
    
    


    return request
