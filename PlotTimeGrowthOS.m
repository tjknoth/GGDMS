% PlotTimeGrowthOS
% this function makes the plots depicting the time growth with the
% distribution of order statistics for floats and doubles.

% use the function todaystring to get today's date in YYYYMMDD
% or manually enter a date string in YYYYMMDD format to select .csv files
filedate = todaystring;      % filedate = '20150128'

fontsz = 16;      % font size for legend and title text

fileprefix = 'Summary';
type=cell(2,1); type{1}='F'; type{2}='D';
typstr=cell(2,1); typstr{1}='Floats'; typstr{2}='Doubles';
vec=cell(4,1); vec{1}='U'; vec{2}='N'; vec{3}='H'; vec{4}='C';
OS=cell(5,1); OS{1}='U'; OS{2}='R'; OS{3}='N'; OS{4}='C'; OS{5}='S';
bms='bucketMultiselect';
legtxt=cell(5,1);
legtxt{5}=['Uniform - ' bms]; 
legtxt{4}=['Uniform Random - ' bms]; 
legtxt{3}=['Normal Random - ' bms]; 
legtxt{2}=['Clustered - ' bms]; 
legtxt{1}=['Sectioned - ' bms]; 
n=2^26;
OSlist=100:10:500;
clist='rgbcm';


for t=1:2
    figure(t)
    hold off
    titlestr=sprintf('Time Growth with Distribution of Order Statistics, Uniform %s, n=2 ^{26}', typstr{t});
    pname=['TimeGrowthDistr' typstr{t} '.pdf'];
    for s=5:-1:1
        filesuffix = [type{t} vec{1} OS{s} filedate];
        fname = [fileprefix filesuffix '.csv'];
        data=csvread(fname);
        data=data((data(:,1)==n),:);
        data=data(ismember(data(:,2),OSlist),:);
        % compute modified mean (exclude min and max times)
        data(:,3) = ( data(:,3).*data(:,11) - data(:,4) - data(:,5) ) ./ ( data(:,11)-2 );
        data(:,7) = ( data(:,7).*data(:,11) - data(:,8) - data(:,9) ) ./ ( data(:,11)-2 );
        if (s==5)
            hold off
            plot(data(:,2), data(:,3), '--k.', 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 2)
        end
        hold on
        line=['-' clist(s) 's'];
        plot(data(:,2), data(:,7), line, 'MarkerFaceColor', clist(s), 'LineWidth', 2, 'MarkerSize', 2)
    end
    lgd=legend('sort&choose',legtxt{1},legtxt{2},legtxt{3},legtxt{4},legtxt{5});
    set(lgd, 'fontsize', fontsz);
    xlabel('number of order statistics','fontsize',fontsz);
    ylabel('milliseconds','fontsize',fontsz);

    %axis([21 26 0 100]);
    %set(gca,'XTick',21:26)

    title(titlestr,'fontsize',fontsz);
    print('-dpdf',pname);
end


