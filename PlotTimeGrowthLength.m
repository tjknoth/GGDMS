% PlotTimeGrowthLength
% this function makes the plots depicting the time growth with the
% length of the vector.

% use the function todaystring to get today's date in YYYYMMDD
% or manually enter a date string in YYYYMMDD format to select .csv files
filedate = todaystring;      % filedate = '20150128'

fontsz = 16;      % font size for legend and title text

fileprefix = 'Summary';
type=cell(3,1); type{1}='F'; type{2}='D'; type{3}='U';
typstr=cell(3,1); typstr{1}='Floats'; typstr{2}='Doubles'; typstr{3}='Uints';
vec=cell(4,1); vec{1}='U'; vec{2}='N'; vec{3}='H'; vec{4}='C';
vecstr=cell(4,1); vecstr{1}='Uniform'; vecstr{2}='Normal'; vecstr{3}='Half Normal'; vecstr{4}='Cauchy';
OS=cell(5,1); OS{1}='U'; OS{2}='R'; OS{3}='N'; OS{4}='C'; OS{5}='S';
bms='bucketMultiselect';
sc='sort&choose';
legtxt=cell(4,1);
% legtxt{5}=['Uniform - ' bms]; 
% legtxt{4}=['Uniform Random - ' bms]; 
% legtxt{3}=['Normal Random - ' bms]; 
% legtxt{2}=['Clustered - ' bms]; 
% legtxt{1}=['Sectioned - ' bms]; 
p=20:29;
nlist=2.^p
numOS=101;
OSlist=100:10:500;
%clist='rgbcm';


for v=1:1
    figure(v)
    hold off
    titlestr=sprintf('101 Percentile Order Statistics, Vector distribution: %s, n=2 ^{26}', vecstr{v});
    pname=['TimeGrowthLengthVecDist' vec{v} '.pdf'];
    for t=1:3
        filesuffix = [type{t} vec{v} OS{1} filedate];
        fname = [fileprefix filesuffix '.csv'];
        data=csvread(fname);
        data=data(ismember(data(:,1),nlist),:);
        data=data((data(:,2)==numOS),:);
        % compute modified mean (exclude min and max times)
        data(:,3) = ( data(:,3).*data(:,11) - data(:,4) - data(:,5) ) ./ ( data(:,11)-2 );
        data(:,7) = ( data(:,7).*data(:,11) - data(:,8) - data(:,9) ) ./ ( data(:,11)-2 );
        data=log2(data);
        size(data)
        if strcmp(type{t},'F')
            line='--';
        elseif strcmp(type{t},'U')
            line='-.';
        else line='-';
        end
        plot(data(:,1), data(:,3), [line 'k.'], 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+1}=[typstr{t} ' - ' sc];
        hold on
        plot(data(:,1), data(:,7), [line 'rs'], 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+2}=[typstr{t} ' - ' bms];
    end
    lgd=legend(legtxt{1},legtxt{2},legtxt{3},legtxt{4},legtxt{5},legtxt{6}, 'Location','NorthWest');
    set(lgd, 'fontsize', fontsz);
    xlabel('log_2(vector length)','fontsize',fontsz);
    ylabel('log_2(milliseconds)','fontsize',fontsz);

    %axis([100 500 0 525]);
    %set(gca,'XTick',21:26)

    title(titlestr,'fontsize',fontsz);
    print('-dpdf',pname);
end

for v=2:2
    figure(v)
    hold off
    titlestr=sprintf('101 Percentile Order Statistics, Vector distribution: %s, n=2 ^{26}', vecstr{v});
    pname=['TimeGrowthLengthVecDist' vec{v} '.pdf'];
    for t=1:2
        filesuffix = [type{t} vec{v} OS{1} filedate];
        fname = [fileprefix filesuffix '.csv'];
        data=csvread(fname);
        data=data(ismember(data(:,1),nlist),:);
        data=data((data(:,2)==numOS),:);
        % compute modified mean (exclude min and max times)
        data(:,3) = ( data(:,3).*data(:,11) - data(:,4) - data(:,5) ) ./ ( data(:,11)-2 );
        data(:,7) = ( data(:,7).*data(:,11) - data(:,8) - data(:,9) ) ./ ( data(:,11)-2 );
        data=log2(data);
        size(data)
        if strcmp(type{t},'F')
            line='--';
        elseif strcmp(type{t},'U')
            line='-.';
        else line='-';
        end
        plot(data(:,1), data(:,3), [line 'k.'], 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+1}=[typstr{t} ' - ' sc];
        hold on
        plot(data(:,1), data(:,7), [line 'rs'], 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+2}=[typstr{t} ' - ' bms];
    end
    lgd=legend(legtxt{1},legtxt{2},legtxt{3},legtxt{4},'Location','NorthWest');
    set(lgd, 'fontsize', fontsz);
    xlabel('log_2(vector length)','fontsize',fontsz);
    ylabel('log_2(milliseconds)','fontsize',fontsz);

    %axis([100 500 0 525]);
    %set(gca,'XTick',21:26)

    title(titlestr,'fontsize',fontsz);
    print('-dpdf',pname);
end

for v=3:4
    figure(v)
    hold off
    titlestr=sprintf('101 Percentile Order Statistics, Vector distribution: %s, n=2 ^{26}', vecstr{v});
    pname=['TimeGrowthLengthVecDist' vec{v} '.pdf'];
    for t=1:1
        filesuffix = [type{t} vec{v} OS{1} filedate];
        fname = [fileprefix filesuffix '.csv'];
        data=csvread(fname);
        data=data(ismember(data(:,1),nlist),:);
        data=data((data(:,2)==numOS),:);
        % compute modified mean (exclude min and max times)
        data(:,3) = ( data(:,3).*data(:,11) - data(:,4) - data(:,5) ) ./ ( data(:,11)-2 );
        data(:,7) = ( data(:,7).*data(:,11) - data(:,8) - data(:,9) ) ./ ( data(:,11)-2 );
        data=log2(data);
        if strcmp(type{t},'F')
            line='--';
        elseif strcmp(type{t},'U')
            line='-.';
        else line='-';
        end
        plot(data(:,1), data(:,3), [line 'k.'], 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+1}=[typstr{t} ' - ' sc];
        hold on
        plot(data(:,1), data(:,7), [line 'rs'], 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+2}=[typstr{t} ' - ' bms];
    end
    lgd=legend(legtxt{1},legtxt{2},'Location','NorthWest');
    set(lgd, 'fontsize', fontsz);
    xlabel('log_2(vector length)','fontsize',fontsz);
    ylabel('log_2(milliseconds)','fontsize',fontsz);

    %axis([100 500 0 525]);
    %set(gca,'XTick',21:26)

    title(titlestr,'fontsize',fontsz);
    print('-dpdf',pname);
end
