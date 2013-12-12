function line = getline(mt, str, stp, width)



if str==stp
    line = mt(:,str);
else

	 length = (size(mt,1)^2 + abs(str-stp)^2)^(1/2);
	 length = round(length);

	 m = (size(mt,1) - 1)/(stp - str);
	 b = 0.501 - ((str - 0.5) * m);

	 for i = 1:1:length

	 x = (str - 0.5) + (i-1) * (stp-str)/(length - 1);
	 y = m * x + b;
	 line(i) = mt(round(y), round(x));
	 for j=1:1:width
	
		 line(i) = line(i) + mt(round(y), min([(round(x) + j) size(mt,2)]) );
		 line(i) = line(i) + mt(round(y), max([(round(x) - j) 1]) );
	 end

	 round(y);
	 round(x);

	 end

end
