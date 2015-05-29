load timerresults11264.mat;

f_timers = subroutTimers{1};
d_timers = subroutTimers{2};

numTests = 25;
OSlist = [11 101 1001];
p=26:2:28;
nlist = 2.^p;

meantimes = zeros(2*length(nlist)*length(OSlist),6);
mintimes = meantimes;
maxtimes = meantimes;

for type=1:2
  timers=subroutTimers{type};
  for n = 1:length(nlist)
    for j=1:length(OSlist)
      currentrows = (n-1)*numTests*length(OSlist) + (j-1)*25 + [1:25];
      data=timers(currentrows,:);
      mins = min(data);
      maxs = max(data);
      means = mean(data);
%size(mins), size(maxs), size(means)
      % truncated mean for smoothing
      means = (means*numTests - mins - maxs)/(numTests-2);
      saverow = ((type-1)*length(nlist) + (n-1))*length(OSlist) + j;
      meantimes(saverow,:)=means;
      mintimes(saverow,:)=mins;
      maxtimes(saverow,:)=maxs;
    end
  end
end

lengths = nlist
orderStats = OSlist
averages = meantimes
minimums = mintimes
maximums = maxtimes


