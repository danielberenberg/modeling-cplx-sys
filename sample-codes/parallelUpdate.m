function [] = parallelUpdate()

plotVEG = 1;
plotFIRE = 1;

n = 256; % grid size
pGrass = 0.001; % p(emptry -> grass)
pLightning = 0.00001;
pGrassFireSpread = 1.0;
pTreeFireSpread = 0.0;

beta = 0.00000;
pTreeSpread = 0.00;

maxTime=2500;

veg = randi([0,2], n,n);
veg(:,:) = 0;

%% visualize the simulation figure(1);
%imh = image(cat(3,z,veg*.2,z)); set(imh, 'erasemode', 'none')
axis equal
axis tight
axis square

clims = [0 3];
cmap = [1,1,1;0.4,0.8,0;0,0.6,0;0.8,0.2,0.2];

T = zeros(1,maxTime);
G = zeros(1,maxTime);
saveT = 0.1;

AllFireSizes = [1, 1];

for t=1:maxTime
    
    %grow some grass
    veg = veg + ((veg==0) & rand(n,n)<pGrass);
    
    %grow spreading trees
    treeSpread = (veg(1:n,[n 1:n-1])==2 & rand(n,n)<pTreeSpread) + (veg(1:n,[2:n 1])==2 & rand(n,n)<pTreeSpread) + (veg([n 1:n-1], 1:n)==2 & rand(n,n)<pTreeSpread) + (veg([2:n 1],1:n)==2 & rand(n,n)<pTreeSpread);
    veg(treeSpread>0) = 2;
    
    %grow seed dispersed trees
    veg(rand(n,n)<beta) = 2;
    
    %light fires (only in grass)
    fire = (veg==1 & (rand(n,n)<pLightning));   
    areaBurning = sum(sum(fire));%this is how many cells are on fire
    Firesize = areaBurning;
    veg(fire==1) = 3;
    
    %while there is any fire, propagate fire
    while(areaBurning>0)
        fireSpread = (fire(1:n,[n 1:n-1])==1 & veg==1 & rand(n,n)<pGrassFireSpread) + ...
                     (fire(1:n,[2:n 1])==1 & veg==1 & rand(n,n)<pGrassFireSpread) + ...
                     (fire([n 1:n-1], 1:n)==1 & veg==1 & rand(n,n)<pGrassFireSpread) + ...
                     (fire([2:n 1],1:n)==1 & veg==1 & rand(n,n)<pGrassFireSpread);
        
        fireSpread = fireSpread + ...
        (fire(1:n,[n 1:n-1])==1 & veg==2 & rand(n,n)<pTreeFireSpread) + ...
        (fire(1:n,[2:n 1])==1 & veg==2 & rand(n,n)<pTreeFireSpread) + ...
        (fire([n 1:n-1], 1:n)==1 & veg==2 & rand(n,n)<pTreeFireSpread) + ...
        (fire([2:n 1],1:n)==1 & veg==2 & rand(n,n)<pTreeFireSpread);
    
        fire = fireSpread>0;
        areaBurning = sum(sum(fire));
        Firesize = Firesize + areaBurning;
        veg(fire==1) = 3;
        
        %%set(imh, 'cdata', cat(3,(veg==1),(veg==2),z) )
        if(plotVEG)
            if(plotFIRE)
                figure(1)
                imagesc(veg,clims); colormap(cmap);
                drawnow
            end
        end
    end

    AllFireSizes = [AllFireSizes, Firesize];
    
    %%set(imh, 'cdata', cat(3,(veg==1),(veg==2),z) )
    if(plotVEG)
        figure(1)
        imagesc(veg,clims); colormap(cmap);
        figure(2)
        loglog(hist(AllFireSizes,1:100),'k*-');
        drawnow
    end
    
    %turn fire to ash
    veg(veg==3) = 0;
    
    T(t) = 100*sum(sum(veg==2))/(n^2);
    G(t) = sum(sum(veg==1))/(n^2);
    %figure(33)
    %hold off
    %plot(T(1:t),'k*')
    %hold on
    %plot(G(1:t))

    if(mod(t,5)==0)
	T(t);
    end
    
    if(T(t)==100)
        t=maxTime;
    end
    
end

save('./output2/LastMatrix.dat', '-ascii','veg');
