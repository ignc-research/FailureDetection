% Auswertung Standmenge Messung vom 31.01.2015 %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Datenpfad
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DatenPfad = '/mnt/daten/DiaTherm/2015-03-31_MyWen_DiaTherm_INA_Laser_Standmenge/data/';
%ResultsPfad = '/mnt/daten/DiaTherm/2015-03-31_MyWen_DiaTherm_INA_Laser_Standmenge/results/';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filenames und Definitionen
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dateinamen für die gemessenen Daten 
%cd('welding data')
% data.filename{1} =  'film1';
% data.filename{2} =  'film2';
% data.filename{3} =  'film3';
% data.filename{4} =  'film4';
% data.filename{5} =  'Probe_1514.tss';
% data.filename{6} =  'Probe_1515.tss';
% data.filename{7} =  'Probe_1516.tss';
% data.filename{8} =  'Probe_1517.tss';
% data.filename{9} =  'Probe_1518.tss';
% data.filename{10} = 'Probe_1519.tss';
% data.filename{11} = 'Probe_1520.tss';
% data.filename{12} =  'Probe_1521.tss';
% 
% data.filename{13} =  'Probe_1522.tss';
% data.filename{14} =  'Probe_1523.tss';
% data.filename{15} =  'Probe_1524.tss';
% data.filename{16} =  'Probe_1525.tss';
% data.filename{17} =  'Probe_1526.tss';
% data.filename{18} =  'Probe_1527.tss';
% data.filename{19} =  'Probe_1528.tss';
% data.filename{20} =  'Probe_1529.tss';
% data.filename{21} =  'Probe_1530.tss';
% data.filename{22} =  'Probe_1531.tss';
% data.filename{23} =  'Probe_1532.tss';
% data.filename{24} =  'Probe_1533.tss';
% 
% data.filename{25} =  'Probe_1544.tss';
% data.filename{26} =  'Probe_1545.tss';
% data.filename{27} =  'Probe_1546.tss';
% data.filename{28} =  'Probe_1547.tss';
% data.filename{29} =  'Probe_1548.tss';
% data.filename{30} =  'Probe_1549.tss';
% data.filename{31} =  'Probe_1550.tss';
% data.filename{32} =  'Probe_1551.tss';
% data.filename{33} =  'Probe_1552.tss';
% data.filename{34} =  'Probe_1553.tss';
% data.filename{35} =  'Probe_1554.tss';
% data.filename{36} =  'Probe_1555.tss';
% 
% data.filename{37} =  'Probe_1556.tss';
% data.filename{38} =  'Probe_1557.tss';
% data.filename{39} =  'Probe_1558.tss';
% data.filename{40} =  'Probe_1559.tss';
% data.filename{41} =  'Probe_1560.tss';
% data.filename{42} =  'Probe_1561.tss';
% data.filename{43} =  'Probe_1562.tss';
% data.filename{44} =  'Probe_1563.tss';
% data.filename{45} =  'Probe_1564.tss';
% data.filename{46} =  'Probe_1565.tss';
% data.filename{47} =  'Probe_1566.tss';
% 
% data.filename{48} =  'Probe_1567.tss';
% data.filename{49} =  'Probe_1568.tss';
% data.filename{50} =  'Probe_1569.tss';
% data.filename{51} =  'Probe_1570.tss';
% data.filename{52} =  'Probe_1571.tss';
% data.filename{53} =  'Probe_1572.tss';
% data.filename{54} =  'Probe_1573.tss';
% data.filename{55} =  'Probe_1574.tss';
% data.filename{56} =  'Probe_1575.tss';
% data.filename{57} =  'Probe_1576.tss';
% data.filename{58} =  'Probe_1577.tss';
% data.filename{59} =  'Probe_1578.tss';
% 
% data.filename{60} =  'Probe_1579.tss';
% data.filename{61} =  'Probe_1580.tss';
% data.filename{62} =  'Probe_1581.tss';
% data.filename{63} =  'Probe_1582.tss';
% data.filename{64} =  'Probe_1583.tss';
% data.filename{65} =  'Probe_1584.tss';
% data.filename{66} =  'Probe_1585.tss';
% data.filename{67} =  'Probe_1586.tss';
% data.filename{68} =  'Probe_1587.tss';
% data.filename{69} =  'Probe_1588.tss';
% data.filename{70} =  'Probe_1589.tss';
% data.filename{71} =  'Probe_1590.tss';
% 
% data.filename{72} =  'Probe_1591.tss';
% data.filename{73} =  'Probe_1592.tss';
% data.filename{74} =  'Probe_1593.tss';
% data.filename{75} =  'Probe_1594.tss';
% data.filename{76} =  'Probe_1595.tss';
% data.filename{77} =  'Probe_1596.tss';
% data.filename{78} =  'Probe_1597.tss';
% data.filename{79} =  'Probe_1598.tss';
% data.filename{80} =  'Probe_1599.tss';
% data.filename{81} =  'Probe_1600.tss';
% data.filename{82} =  'Probe_1601.tss';
% data.filename{83} =  'Probe_1602.tss';
% 
% data.filename{84} =  'Probe_1603.tss';
% data.filename{85} =  'Probe_1604.tss';
% data.filename{86} =  'Probe_1605.tss';
% data.filename{87} =  'Probe_1606.tss';
% data.filename{88} =  'Probe_1607.tss';
% data.filename{89} =  'Probe_1608.tss';
% data.filename{90} =  'Probe_1609.tss';
% data.filename{91} =  'Probe_1610.tss';
% data.filename{92} =  'Probe_1611.tss';
% data.filename{93} =  'Probe_1612.tss';
% data.filename{94} =  'Probe_1613.tss';
% data.filename{95} =  'Probe_1614.tss';
% 
% data.filename{96} =  'Probe_1615.tss';
% data.filename{97} =  'Probe_1616.tss';
% data.filename{98} =  'Probe_1617.tss';
% data.filename{99} =  'Probe_1618.tss';
% data.filename{100} =  'Probe_1619.tss';
% data.filename{101} =  'Probe_1620.tss';
% data.filename{102} =  'Probe_1621.tss';
% data.filename{103} =  'Probe_1622.tss';
% data.filename{104} =  'Probe_1623.tss';
% data.filename{105} =  'Probe_1624.tss';
% data.filename{106} =  'Probe_1625.tss';
% 
% data.filename{107} =  'Probe_1626.tss';
% data.filename{108} =  'Probe_1627.tss';
% data.filename{109} =  'Probe_1628.tss';
% data.filename{110} =  'Probe_1629.tss';
% data.filename{111} =  'Probe_1630.tss';
% data.filename{112} =  'Probe_1631.tss';
% data.filename{113} =  'Probe_1632.tss';
% data.filename{114} =  'Probe_1633.tss';
% data.filename{115} =  'Probe_1634.tss';



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i = 1:115
 data.filename{i} = load(['film' num2str(i) '.mat']);
end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Filme einladen und Korrekturen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Alle gemessenen Daten einlesen. Es wird schon ein Subframe benutzt um
% moeglichst wenig Daten einlesen zu muessen

%cd(DatenPfad);

%SubFrame = [60 190 90 235];
%SubFrame = [90 235 60 190];


for n = 1:size(data.filename, 2)
    
    film = load(['film' num2str(n) '.mat']);
    film = cell2mat(struct2cell(film));
    
    %[Max_Wert, Max_FrameNo] = max(squeeze(film(65, 73, :)));
    Max_FrameNo = 40;
    % RawImage zur Mitte der Strahlzeit (Strahlzeit 40 Frames)
    
    frame_OhneBGC = film(:,:, Max_FrameNo - 20);
    
    % Plots zeitlicher Verlauf %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %figure(1); 
    %set(figure(1), 'Position', [1 1 1200 650]);
   
    %plot(squeeze(film(65, 72, :)));
    %LegendString = strcat(data.filename{n}(1:end-4));
   
    %h = legend(LegendString); 
    %set(h, 'interpreter', 'none', 'Location', 'northwest', 'FontSize', 12);
   
    % Speichern
%    export_fig(strcat(data.filename{n}, '-T-vs-t.png'), figure(1), '-painters', '-r300', '-a1');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Nullbildabzug
    
    film = bsxfun(@minus, film, mean(film(:,:,1:3),3));
    
    % Emissivitätskorrektur
    
    film = bsxfun(@rdivide, film, mean(film(:,:, 140:143), 3));
    
    % Plots
    
    hXCFdetrend = figure(2);
    
    set(hXCFdetrend,'Color',[1 1 1],'Tag','Images','HandleVisibility', 'on', 'Position', [1 1 1600 320]);
    subplot(1,4,1);
    imagesc(frame_OhneBGC);
    axis image;
    xlabel('x [pixel]');
    ylabel('y [pixel]');
   % title(['ohne Korrektur - Frame 20', data.filename{n}],'Interpreter','none');
   
    subplot(1,4,2);
    imagesc(film(:,:, Max_FrameNo));
    axis image;
    xlabel('x [pixel]');
    ylabel('y [pixel]');
    %title(['MaxFrameNo', data.filename{n}],'Interpreter','none');
    
    subplot(1,4,3);
    imagesc(film(:,:, Max_FrameNo + 2));
    axis image;
    xlabel('x [pixel]');
    ylabel('y [pixel]');
    %title(['MaxFrameNo + 2', data.filename{n}],'Interpreter','none');
    
    subplot(1,4,4);
    imagesc(film(:,:, Max_FrameNo + 4));
    axis image;
    xlabel('x [pixel]');
    ylabel('y [pixel]');
    %title(['MaxFrameNo + 4', data.filename{n}],'Interpreter','none');
    %save(['normed_data_new/norm_film' num2str(n) '.mat'],'film');
    %clear title; titlename = strcat(data.filename{n});
    %set(figure(2), 'Color', [1 1 1]); 
    
    for j = 91:115
        figure(3)
        imagesc(film(:,:, j)); % disable plotting
        axis image;
        xlabel('x [pixel]');
        ylabel('y [pixel]');
        saveas(gcf,['dataset91/norm_film' num2str(n) '_frame' num2str(j) '.jpg'])
        % Speichern
        %export_fig(strcat(data.filename{n}, '_Frames.png'), figure(2), '-painters', '-r300', '-a1');
        %frames(:,:,n) = frame_OhneBGC;
        %frames_temp(:,:,n) = abs(film(:,:, Max_FrameNo - 20)-max(max(film(:,:, Max_FrameNo - 20))));
        
    end
    
%         figure(3)
%         imagesc(film(:,:, 60));
%         axis image;
%         xlabel('x [pixel]');
%         ylabel('y [pixel]');
%         saveas(gcf,['normed_data_new_jpg/norm_film' num2str(n) '.jpg'])
%         % Speichern
end

clear SubFrame; disp('load data...........done');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
