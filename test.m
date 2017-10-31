tic
for x = 1:41
eval(['range_profile_data',num2str(x), '= [];']);
end
%eval()中是被它操作的字符串，[]中是被拼接起来的字符串
Ni = [323 549 525 712 712 1251 620 536 337 1170 706 317 411 856 433 850 400 326 644 331 832 656 504 667 597 655 483 932 538 485 317 597 542 634 999 384 1252 336 463 262 177];

for Data_block = 1:41;
for i=0:Ni(Data_block)    
    if(Data_block==1)
    range_profile_data1 = [ range_profile_data1 double(h5read('BD_20120310_112546_8126.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==2)
    range_profile_data2 = [ range_profile_data2 double(h5read('BD_20120310_162802_8795.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))]; 
    elseif(Data_block==3)
    range_profile_data3 = [ range_profile_data3 double(h5read('BD_20120311_083424_8008.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==4)
    range_profile_data4 = [ range_profile_data4 double(h5read('BD_20120311_103436_8004.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==5)
    range_profile_data5 = [ range_profile_data5 double(h5read('BD_20120311_104148_8004.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==6)
    range_profile_data6 = [ range_profile_data6 double(h5read('BD_20120311_174522_8784.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==7)
    range_profile_data7 = [ range_profile_data7 double(h5read('BD_20120311_215020_8337.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==8)
    range_profile_data8 = [ range_profile_data8 double(h5read('BD_20120312_105852_8001.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==9)
    range_profile_data9 = [ range_profile_data9 double(h5read('BD_20120313_104652_8001.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==10) 
    range_profile_data10 = [ range_profile_data10 double(h5read('BD_20120313_171200_8001.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==11) 
    range_profile_data11 = [ range_profile_data11 double(h5read('BD_20120313_210448_8010.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==12)
    range_profile_data12 = [ range_profile_data12 double(h5read('BD_20120314_000016_8025.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==13)
    range_profile_data13 = [ range_profile_data13 double(h5read('BD_20120314_074354_8029.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==14)
    range_profile_data14 = [ range_profile_data14 double(h5read('BD_20120314_112030_8031.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==15)
    range_profile_data15 = [ range_profile_data15 double(h5read('BD_20120419_113138_8048.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==16)
    range_profile_data16 = [ range_profile_data16 double(h5read('BD_20120420_181156_8128.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==17)
    range_profile_data17 = [ range_profile_data17 double(h5read('BD_20120420_210609_8035.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==18)
    range_profile_data18 = [ range_profile_data18 double(h5read('BD_20120420_224818_8090.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==19)
    range_profile_data19 = [ range_profile_data19 double(h5read('BD_20120421_052757_8001.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==20) 
    range_profile_data20 = [ range_profile_data20 double(h5read('BD_20120421_115325_8015.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==21) 
    range_profile_data21 = [ range_profile_data21 double(h5read('BD_20120421_180224_8002.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==22)
    range_profile_data22 = [ range_profile_data22 double(h5read('BD_20120421_204204_8008.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==23)
    range_profile_data23 = [ range_profile_data23 double(h5read('BD_20120421_224731_8013.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==24)
    range_profile_data24 = [ range_profile_data24 double(h5read('BD_20120422_102122_8006.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==25)
    range_profile_data25 = [ range_profile_data25 double(h5read('BD_20120422_123723_8015.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==26)
    range_profile_data26 = [ range_profile_data26 double(h5read('BD_20120422_163333_8005.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==27)
    range_profile_data27 = [ range_profile_data27 double(h5read('BD_20120422_211016_8012-300-40-LMA CGM VOLTARE.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==28)
    range_profile_data28 = [ range_profile_data28 double(h5read('BD_20120422_224316_8009-96-16.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==29)
    range_profile_data29 = [ range_profile_data29 double(h5read('BD_20120423_000010_8009.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==30) 
    range_profile_data30 = [ range_profile_data30 double(h5read('BD_20120423_000927_8009.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==31)
    range_profile_data31 = [ range_profile_data31 double(h5read('BD_20120423_093319_8001.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==32)
    range_profile_data32 = [ range_profile_data32 double(h5read('BD_20120423_112654_8014.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==33)
    range_profile_data33 = [ range_profile_data33 double(h5read('BD_20120423_154638_8801.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==34)
    range_profile_data34 = [ range_profile_data34 double(h5read('BD_20120423_173431_8009.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==35)
    range_profile_data35 = [ range_profile_data35 double(h5read('BD_20120423_193623_8010.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==36)
    range_profile_data36 = [ range_profile_data36 double(h5read('BD_20120423_232049_8041.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==37)
    range_profile_data37 = [ range_profile_data37 double(h5read('BD_20120424_093856_8010.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==38)
    range_profile_data38 = [ range_profile_data38 double(h5read('BD_20120429_181552_8001.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==39)
    range_profile_data39 = [ range_profile_data39 double(h5read('BD_20120429_210232_8015.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==40)
    range_profile_data40 = [ range_profile_data40 double(h5read('BD_20120423_133814_8004.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    elseif(Data_block==41)
    range_profile_data41 = [ range_profile_data41 double(h5read('BD_20120424_144936_8012.h5', strcat('/G_',num2str(i),'/CH_0', '/DATA')))];
    end
end

if(Data_block==1)
 range_profile_data1(:,[368,453,539,626,764,927]) = [];
 elseif(Data_block==2)
 range_profile_data2(:,[374,405,573,931,999]) = [];
 elseif(Data_block==3)
 range_profile_data3(:,[1:3,335,364,414,513,735,799,988,1292,1524,1557,1520:1578]) = [];
 elseif(Data_block==4)
 range_profile_data4(:,[522,569,705,818,926,1020,1136,1250,1469,1518,1576,1796,1961]) = [];
 elseif(Data_block==5)
 range_profile_data5(:,[486,533,669,782,984,1035,1100,1214,1433,1482,1540,1760,1925,2038,890,919,916]) = [];
 elseif(Data_block==6)
 range_profile_data6(:,[2209:2216,3433]) = [];
 elseif(Data_block==7)
 range_profile_data7(:,[1321,1322,1323,1075,255,1:3,1048:1050]) = [];
 elseif(Data_block==8)
 range_profile_data8(:,[555:579,754:756,243,739,819,902,102,112,364,439,1608]) = [];
 elseif(Data_block==9)
 range_profile_data9(:,[25:264,286:1014]) = [];
 elseif(Data_block==10)
 range_profile_data10(:,[1097,1443,1499,1508,2601,2708,2742,2971,3298,2985,3370,2170:3502]) = [];
 elseif(Data_block==11)
 range_profile_data11(:,[42,58,525,1298,1370,1371,1466,1980:2121]) = [];
 elseif(Data_block==12)
 range_profile_data12(:,[56,696,759]) = [];
 elseif(Data_block==13)
 range_profile_data13(:,[197,1106,537,838:867]) = [];
 elseif(Data_block==14)
 range_profile_data14(:,[54,362,582,576,1346,1412,870:885,1561:2571]) = [];
 elseif(Data_block==15)
 range_profile_data15(:,[201,436,902,999]) = [];
 elseif(Data_block==16)
 range_profile_data16(:,[151,707,2018:2549]) = [];
 elseif(Data_block==17)
 range_profile_data17(:,[166,380,475,569,390,736:801,940:951,1030:1203]) = [];
 elseif(Data_block==18)
 range_profile_data18(:,[274,368,392,574,970:981]) = [];
 elseif(Data_block==20)
 range_profile_data20(:,[1:3,7:9,140,264,265,286:298,64,75,91,236,270,789,915,942,951,968:996]) = [];
 elseif(Data_block==21)
 range_profile_data21(:,[1138,1256,1841,1993,1965,2391]) = [];
 elseif(Data_block==22)
 range_profile_data22(:,[1776,1783:1785]) = [];
 elseif(Data_block==23)
 range_profile_data23(:,[59,67,128]) = [];
 elseif(Data_block==24)
 range_profile_data24(:,[40,105,344,346,605,732,769,788,848,851,922,1042,1119,1528,1544,1663,1687,258]) = [];
 elseif(Data_block==25)
 range_profile_data25(:,[25,130,132,138,202,308,343,670,862,914,993,1026,1570,1679,1121,961,1329,1608,1588,1741:1794]) = [];
 elseif(Data_block==26)
 range_profile_data26(:,[39,54,122,128,301,311,463,571,778,857,941,1314,1359,552,972,1131,1181,1419,1550,1681]) = [];
 elseif(Data_block==27)
 range_profile_data27(:,[612,173,1183:1197,1363:1365,1249:1452]) = [];
 elseif(Data_block==28)
 range_profile_data28(:,[486,494,1015,670,416,1293,1391,2083]) = [];
 elseif(Data_block==29)
 range_profile_data29(:,[111,209,901,992,1492]) = [];
 elseif(Data_block==30)
 range_profile_data30(:,[50,742,832,833,1333]) = [];
 elseif(Data_block==31)
 range_profile_data31(:,[284,240,20:50,52:65,285:310,402:430,511,655:673,927:954]) = [];
 elseif(Data_block==32)
 range_profile_data32(:,[85,333,444,502,1044,1086,1247,1768,1769]) = [];
 elseif(Data_block==33)
 range_profile_data33(:,[73,392,541,995,1012,1274]) = [];
 elseif(Data_block==34)
 range_profile_data34(:,[187,367,317,377,397,945,1087,1384,1453]) = [];
 elseif(Data_block==35)
 range_profile_data35(:,[437,485,1880,1881,2234,2709,2415,2672]) = [];
 elseif(Data_block==36)
 range_profile_data36(:,[502]) = [];
 elseif(Data_block==37)
 range_profile_data37(:,[227,981,1331,3412,3004,3355,3506,2084,3488,697,2700:3759]) = [];
 elseif(Data_block==38)
 range_profile_data38(:,[760:765]) = [];
 elseif(Data_block==39)
 range_profile_data39(:,[1326:1392]) = [];
 elseif(Data_block==40)
 range_profile_data40(:,[280:292,7,161,166,288,297,354,610,635,726,762:789]) = [];
 elseif(Data_block==41)
 range_profile_data41(:,[1:6,39,78,220,239,294,373,440]) = [];
 end
 
 eval(['[Y,I',num2str(Data_block),'] = max(sum(abs(range_profile_data',num2str(Data_block),').^2,1));']);
 %b=sum(a,dim); a表示矩阵；dim等于1或者2，1表示每一列进行求和，2表示每一行进行求和；表示每列求和还是每行求和；b表示求得的行向量。
 %[Y,I]=max(A)：返回行向量Y和I，Y向量记录A的每列的最大值，I向量记录每列最大值的行号。
 eval(['[Nrange',num2str(Data_block),',Nprofile',num2str(Data_block),'] = size(range_profile_data',num2str(Data_block),');']);
 eval(['range_profile_template',num2str(Data_block),' = range_profile_data',num2str(Data_block),'(:,I',num2str(Data_block),');']);
 
  if(Data_block==8)
  range_profile_template8 = range_profile_data8(:,1100);
   elseif(Data_block==3)
  range_profile_template3 = range_profile_data3(:,500);
  elseif(Data_block==19)
  range_profile_template19 = range_profile_data19(:,1305);
  elseif(Data_block==22)
  range_profile_template22 = range_profile_data22(:,800);
   elseif(Data_block==30)
  range_profile_template30 = range_profile_data30(:,60);
   elseif(Data_block==34)
  range_profile_template34 = range_profile_data34(:,60);
   elseif(Data_block==39)
  range_profile_template39 = range_profile_data39(:,200);
 end

 for nn=1:eval(['Nprofile',num2str(Data_block)])     
     eval(['[Y, xcorr_coef',num2str(Data_block),'] = max(xcorr(range_profile_template',num2str(Data_block),',range_profile_data',num2str(Data_block),'(:,nn)));'])
      %取两列信号中互相关系数最大的时间点为xcorr_coef
      eval(['xcorr_bias',num2str(Data_block),' = xcorr_coef',num2str(Data_block),' - Nrange',num2str(Data_block),';'])
       eval(['range_profile_data',num2str(Data_block),'(:,nn) = circshift(range_profile_data',num2str(Data_block),'(:,nn), [xcorr_bias',num2str(Data_block),' 0]);'])
  %circshift(ref,[a b])a是列向移动,b是行向移动
 end
 
   if(Data_block==6)
   range_profile_data6(:,673:3747) = circshift(range_profile_data6(:,673:3747), [-50 0]);
   elseif(Data_block==3)
   range_profile_data3(:,[1255:1508]) = [];
   elseif(Data_block==7)
   range_profile_data7(:,1085:1098) = circshift(range_profile_data7(:,1085:1098), [-20 0]);
    elseif(Data_block==8)
   range_profile_data8(:,[50:340]) = circshift(range_profile_data8(:,[50:340]), [-170 0]);
  elseif(Data_block==19)
   [m,n] = find(sum(abs(range_profile_data19).^2,1)<5e+20 & sum(abs(range_profile_data19).^2,1)>5e+19);
   range_profile_data19 = range_profile_data19(:,n);
range_profile_data19(:,[1:67,166:174,326:352,201,205,208]) = [];
  elseif(Data_block==26)
  range_profile_data26(:,1267:1948) = circshift(range_profile_data26(:,1267:1948), [-100 0]);   range_profile_data26(:,[1240:1280]) = [];
   end
   
   if(Data_block==1)
 range_profile_data1 = range_profile_data1(2401:3500,:);
 elseif(Data_block==2)
 range_profile_data2 = range_profile_data2(1751:2850,:);
 elseif(Data_block==3)
 range_profile_data3 = range_profile_data3(1651:2750,:);
 elseif(Data_block==4)
 range_profile_data4 = range_profile_data4(2051:3150,:);
 elseif(Data_block==5)
 range_profile_data5 = range_profile_data5(2001:3100,:);
 elseif(Data_block==6)
 range_profile_data6 = range_profile_data6(2501:3600,:);
 elseif(Data_block==7)
 range_profile_data7 = range_profile_data7(2301:3400,:);
 elseif(Data_block==8)
range_profile_data8 = range_profile_data8(1851:2950,:);
 elseif(Data_block==9)
range_profile_data9 = range_profile_data9(1:1100,:);
 elseif(Data_block==10)
range_profile_data10 = range_profile_data10(1451:2550,:);
 elseif(Data_block==11)
 range_profile_data11 = range_profile_data11(2101:3200,:);
 elseif(Data_block==12)
 range_profile_data12 = range_profile_data12(1901:3000,:);
 elseif(Data_block==13)
range_profile_data13 = range_profile_data13(2051:3150,:);
 elseif(Data_block==14)
 range_profile_data14 = range_profile_data14(1651:2750,:);
 elseif(Data_block==15)
range_profile_data15 = range_profile_data15(1051:2150,:);
 elseif(Data_block==16)
range_profile_data16 = range_profile_data16(1801:2900,:);
 elseif(Data_block==17)
range_profile_data17 = range_profile_data17(1651:2750,:);
 elseif(Data_block==18)
range_profile_data18 = range_profile_data18(1851:2950,:);
 elseif(Data_block==19)
range_profile_data19 = range_profile_data19(1001:2100,:);
 elseif(Data_block==20)
range_profile_data20 = range_profile_data20(851:1950,:);
 elseif(Data_block==21)
range_profile_data21 = range_profile_data21(851:1950,:);
 elseif(Data_block==22)
range_profile_data22 = range_profile_data22(1951:3050,:);
 elseif(Data_block==23)
range_profile_data23 = range_profile_data23(1851:2950,:);
 elseif(Data_block==24)
range_profile_data24 = range_profile_data24(651:1750,:);
 elseif(Data_block==25)
range_profile_data25 = range_profile_data25(701:1800,:);
 elseif(Data_block==26)
range_profile_data26 = range_profile_data26(801:1900,:);
 elseif(Data_block==27)
range_profile_data27 = range_profile_data27(1001:2100,:);
 elseif(Data_block==28)
range_profile_data28 = range_profile_data28(801:1900,:);
 elseif(Data_block==29)
range_profile_data29 = range_profile_data29(801:1900,:);
 elseif(Data_block==30)
range_profile_data30 = range_profile_data30(851:1950,:);
 elseif(Data_block==31)
range_profile_data31 = range_profile_data31(551:1650,:);
 elseif(Data_block==32)
range_profile_data32 = range_profile_data32(2251:3350,:);
 elseif(Data_block==33)
range_profile_data33 = range_profile_data33(1651:2750,:);
 elseif(Data_block==34)
range_profile_data34 = range_profile_data34(1651:2750,:);
 elseif(Data_block==35)
range_profile_data35 = range_profile_data35(1801:2900,:);
 elseif(Data_block==36)
range_profile_data36 = range_profile_data36(1501:2600,:);
 elseif(Data_block==37)
range_profile_data37 = range_profile_data37(1551:2650,:);
 elseif(Data_block==38)
range_profile_data38 = range_profile_data38(901:2000,:);
 elseif(Data_block==39)
range_profile_data39 = range_profile_data39(801:1900,:);
 elseif(Data_block==40)
range_profile_data40 = range_profile_data40(1751:2850,:);
 elseif(Data_block==41)
range_profile_data41 = range_profile_data41(1651:2750,:);
   end
  %figure, eval(['imagesc(range_profile_data',num2str(Data_block),');'])
  eval(['[Mrange',num2str(Data_block),',Mprofile',num2str(Data_block),'] = size(range_profile_data',num2str(Data_block),');']);
  eval(['M',num2str(Data_block), '= Data_block*ones(1,Mprofile',num2str(Data_block),')']);
end
ship_data = [range_profile_data1 range_profile_data2 range_profile_data3 range_profile_data4 range_profile_data5 range_profile_data6...
    range_profile_data7 range_profile_data8 range_profile_data9 range_profile_data10 range_profile_data11 range_profile_data12 range_profile_data13...
    range_profile_data14 range_profile_data15 range_profile_data16 range_profile_data17 range_profile_data18 range_profile_data19 range_profile_data20...
    range_profile_data21 range_profile_data22 range_profile_data23 range_profile_data24 range_profile_data25 range_profile_data26 range_profile_data27...
    range_profile_data28 range_profile_data29 range_profile_data30 range_profile_data31 range_profile_data32 range_profile_data33 range_profile_data34 ...
    range_profile_data35 range_profile_data36 range_profile_data37 range_profile_data38 range_profile_data39 range_profile_data40 range_profile_data41];
ship_data = mapminmax(ship_data,0,1);
ship_target = [M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 M27 M28 M29 M30 M31 M32 M33 M34 M35 M36 M37 M38 M39 M40 M41];

time = toc
