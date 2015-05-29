% SMOStimingsTable.m
% This function creates the tables of mean timings and acceleration ratio

% use the function todaystring to get today's date in YYYYMMDD
% or manually enter a date string in YYYYMMDD format to select .csv files
filedate = todaystring;      % filedate = '20150128'
filedate = '20150214';

fileprefix = 'Summary';
type=cell(3,1); type{1}='F'; type{2}='D'; type{3}='U';
typstr=cell(3,1); typstr{1}='Floats'; typstr{2}='Doubles'; typstr{3}='Uints';
vec=cell(4,1); vec{1}='U'; vec{2}='N'; vec{3}='H'; vec{4}='C';
vecstr=cell(4,1); vecstr{1}='Uniform'; vecstr{2}='Normal'; vecstr{3}='Half Normal'; vecstr{4}='Cauchy';
OS=cell(5,1); OS{1}='U'; OS{2}='R'; OS{3}='N'; OS{4}='C'; OS{5}='S';

p=24:2:28;
nlist=2.^p;
OSlist=[5, 11, 101, 1001];
OSdistr=OS{1};

data=cell(3,4);

for v=1:4
    vecdistr=vec{v};
    switch v
        case 1
            for t=1:3
                vectype=type{t};
                filesuffix = [vectype vecdistr OSdistr filedate];
                fname = [fileprefix filesuffix '.csv'];
                temp=csvread(fname);
                temp=temp(ismember(temp(:,1),nlist),:);
                temp=temp(ismember(temp(:,2),OSlist),:);
                % compute modified mean (exclude min and max times)
                temp(:,3) = ( temp(:,3).*temp(:,11) - temp(:,4) - temp(:,5) ) ./ ( temp(:,11)-2 );
                temp(:,7) = ( temp(:,7).*temp(:,11) - temp(:,8) - temp(:,9) ) ./ ( temp(:,11)-2 );
                data{t,v}=temp;
            end
        case 2
            for t=1:2
                vectype=type{t};
                filesuffix = [vectype vecdistr OSdistr filedate];
                fname = [fileprefix filesuffix '.csv'];
                temp=csvread(fname);
                temp=temp(ismember(temp(:,1),nlist),:);
                temp=temp(ismember(temp(:,2),OSlist),:);
                % compute modified mean (exclude min and max times)
                temp(:,3) = ( temp(:,3).*temp(:,11) - temp(:,4) - temp(:,5) ) ./ ( temp(:,11)-2 );
                temp(:,7) = ( temp(:,7).*temp(:,11) - temp(:,8) - temp(:,9) ) ./ ( temp(:,11)-2 );
                data{t,v}=temp;
            end
        otherwise
            t=1;
            vectype=type{t};
            filesuffix = [vectype vecdistr OSdistr filedate];
            fname = [fileprefix filesuffix '.csv'];
            temp=csvread(fname);
            temp=temp(ismember(temp(:,1),nlist),:);
            temp=temp(ismember(temp(:,2),OSlist),:);
            % compute modified mean (exclude min and max times)
            temp(:,3) = ( temp(:,3).*temp(:,11) - temp(:,4) - temp(:,5) ) ./ ( temp(:,11)-2 );
            temp(:,7) = ( temp(:,7).*temp(:,11) - temp(:,8) - temp(:,9) ) ./ ( temp(:,11)-2 );
            data{t,v}=temp;
    end
end


% create a .tex file containing the table
texfilename = 'SMOStimingsTable.tex';
fid=fopen(texfilename,'wt');



fprintf('\n\nLatex table....\n\n');

tmp=sprintf('\\\\begin{table}\\\\centering\n');
fprintf(fid,tmp);
tmp=sprintf(['\\\\tbl{Mean timings and acceleration ratios for selecting quartiles, deciles, percentiles, and $1/10$-percentiles, K40.\\\\label{tab:timings}}{%%\n']);
fprintf(fid,tmp);
tmp=sprintf('\\\\begin{tabular}{c} \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\begin{tabular}{||c|r||ccc|ccc||}\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Type} & \\\\multicolumn{6}{|c||}{Float}  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Distribution} & \\\\multicolumn{3}{|c|}{Uniform} & \\\\multicolumn{3}{|c||}{Normal} \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('length & \\\\#OS  & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$ & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$ \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);




    for i=1:length(nlist)*length(OSlist)
        tmp=sprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f  \\\\\\\\ \n', ...
            log2(data{1,1}(i,1)), data{1,1}(i,2), data{1,1}(i,7), data{1,1}(i,3), data{1,1}(i,3)/data{1,1}(i,7), ...
            data{1,2}(i,7), data{1,2}(i,3), data{1,2}(i,3)/data{1,2}(i,7));
        fprintf(fid,tmp);
    end

tmp=sprintf('\\\\hline \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{tabular}  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\begin{tabular}{||c|r||ccc|ccc||}\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Distribution} & \\\\multicolumn{3}{|c|}{Half Normal} & \\\\multicolumn{3}{|c||}{Cauchy} \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('length & \\\\#OS  & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$ & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);


    for i=1:length(nlist)*length(OSlist)
        tmp=sprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f  \\\\\\\\ \n', ...
            log2(data{1,3}(i,1)), data{1,3}(i,2), data{1,3}(i,7), data{1,3}(i,3), data{1,3}(i,3)/data{1,3}(i,7), ...
            data{1,4}(i,7), data{1,4}(i,3), data{1,4}(i,3)/data{1,4}(i,7));
        fprintf(fid,tmp);
    end
    
    
tmp=sprintf('\\\\hline \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{tabular} \\\\\\\\ \n');
fprintf(fid,tmp);

%tmp=sprintf(' \\\\\\\\ \n');  % This will insert a space between the floats and doubles tables.
%fprintf(fid,tmp);
    
tmp=sprintf('\\\\begin{tabular}{||c|r||ccc|ccc||}\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Type} & \\\\multicolumn{6}{|c||}{Double}  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Distribution} & \\\\multicolumn{3}{|c|}{Uniform} & \\\\multicolumn{3}{|c||}{Normal} \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('length & \\\\#OS  & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$ & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);


    for i=1:length(nlist)*length(OSlist)
        tmp=sprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f  \\\\\\\\ \n', ...
            log2(data{2,1}(i,1)), data{2,1}(i,2), data{2,1}(i,7), data{2,1}(i,3), data{2,1}(i,3)/data{2,1}(i,7), ...
            data{2,2}(i,7), data{2,2}(i,3), data{2,2}(i,3)/data{2,2}(i,7));
        fprintf(fid,tmp);
    end
    
    
tmp=sprintf('\\\\hline \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{tabular} \\\\\\\\ \n');
fprintf(fid,tmp);
    
%tmp=sprintf(' \\\\\\\\ \n');   % This will insert a space between the doubles and uint tables.
%fprintf(fid,tmp);

tmp=sprintf('\\\\begin{tabular}{||c|r||ccc||}\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Type} & \\\\multicolumn{3}{|c||}{Unsigned Integers}  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('  \\\\multicolumn{2}{||r||}{Vector Distribution} & \\\\multicolumn{3}{|c||}{Uniform}  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);
tmp=sprintf('length & \\\\#OS  & \\\\texttt{bMS} & \\\\texttt{s\\\\&c} & $\\\\frac{\\\\texttt{s\\\\&c}}{\\\\texttt{bMS}}$  \\\\\\\\ \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\hline\n');
fprintf(fid,tmp);




    for i=1:length(nlist)*length(OSlist)
        tmp=sprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f  \\\\\\\\ \n', ...
            log2(data{3,1}(i,1)), data{3,1}(i,2), data{3,1}(i,7), data{3,1}(i,3), data{3,1}(i,3)/data{3,1}(i,7));
        fprintf(fid,tmp);
    end
    
    
tmp=sprintf('\\\\hline \n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{tabular}\n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{tabular}}\n');
fprintf(fid,tmp);
tmp=sprintf('\\\\end{table}\n');
fprintf(fid,tmp);

fclose(fid);

fprintf('\n\nLatex table saved as %s \n\n', texfilename); 
    






