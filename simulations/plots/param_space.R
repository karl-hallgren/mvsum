
library(RColorBrewer)
library(tikzDevice)
library(scales)

library(latex2exp)


shades <- brewer.pal(10+1, 'RdYlGn')


ns <- c(200, 400, 800) # c(200, 500, 1000)
thetas <- c(0.4, 0.8)


#  pwd = '/Users/karlh/Documents/PhD/movingsum/code_git/simulations/'

pwd = 'simulations/'



#pdf(paste0(pwd,"plots/ParamSpace.pdf"), height=7*0.8, width=10*0.8)

pdf(paste0(pwd,"plots/ParamSpace.pdf"), height=7.5*0.6, width=10*0.6)


par(mfrow=c(2,3))


for(j in 1:2){
  for(i in 1:3){
    
    n <- ns[i]
    theta <- thetas[j]
    
    results <- read.csv(paste0(pwd, 'results/pspace_ind_N', n, '_theta', theta*10,'.csv') )
    results <- as.matrix( results[, 2:length(results)] )
    
    if( j ==1 ){
      if( i == 1){
        par(mar=c(0.7, 3.7, 3.2, 0.2))
      } else if ( i==2 ){
        par(mar=c(0.7, 1.8, 3.2, 1.9))
      } else{
        par(mar=c(0.7, 0.1, 3.2, 3.8))
      }
      
    } else {
      
      if( i == 1){
        par(mar=c(3.4, 3.7, 0.5, 0.2))
      } else if ( i==2 ){
        par(mar=c(3.4, 1.8, 0.5, 1.9))
      } else{
        par(mar=c(3.4, 0.1, 0.5, 3.8))
      }
    }
    
    
    image( t(results),
           axes = FALSE,
           ylab = ifelse(i==1 ,TeX( paste0("$m^{/}$, with $\\theta =",theta, "$") ), '' ),
           xlab = ifelse( j==2 , TeX(paste0("$m$, with $n=", n, "$")), ''),
           cex = 2,#0.8,
           cex.lab = 1.5,
           mgp = c(2.4, 2.7, 1),
           cex.axis = 5,#0.85,
           col = shades#,
           #main = paste("n = ", n)
    )
    box()
    axis(1, at  = seq(0,1, length.out = 10), #cex = 2,
         labels = ifelse( rep(j==2, 10) , round( seq(0, 30, length.out = 10) ) , rep(' ', 10) )
    )
    axis(2, at  = seq(0,1, length.out = 10), #cex = 2,
         labels = ifelse( rep(i==1, 10) , round( seq(0, 30, length.out = 10) ), rep(' ', 10))
    )
    
    if (i == 2 &j == 1){
      par(xpd=TRUE)
      legend("top",
             inset=c(-1.0,-0.12), #c(-0.2,-0.2)
             legend= c('0.0','0.2','0.4','0.6','0.8','1.0'),#round( as.numeric(dimnames(imgaggYjump)[[2]] ),3) ,  #sort( unique( as.numeric(imgaggYjump) ) ) ,
             bty = "n",
             cex = 0.8,
             horiz = TRUE,
             #title="Estimated probability",
             fill = brewer.pal(6, "RdYlGn")
      )
      par(xpd=FALSE)
      
    }
    
  }
}




dev.off()







