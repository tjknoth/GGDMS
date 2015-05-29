function today = todaystring()

datevector=round(clock);

yy=num2str(datevector(1));
mm=num2str(datevector(2));
dd=num2str(datevector(3));

if (length(mm)<2)
  mm=['0' mm];
end
if (length(dd)<2)
  dd=['0' dd];
end

today = [yy mm dd];
