film = load('film5.mat');
film = cell2mat(struct2cell(film));

intensity = 0;
x = linspace(1,249,249);
   for j = 1:249
        intensity(j) = max(squeeze(max(film(:,:, j))));
   end

plot(x,intensity);
%class(intensity)
%disp(intensity)

fidtrg = fopen('intensity_5.txt' ,'wt');
a = num2str(intensity);
fprintf(fidtrg, '%s',a);
fclose(fidtrg);