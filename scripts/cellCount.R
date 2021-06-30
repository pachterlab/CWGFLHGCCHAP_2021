library(ggplot2)

forPlot <- data.frame(
  y = c(23.5,25.5,25,27.5,29.5,30,33.5,28.5,13,13.5,11,10.5,11.5,12.5,11.5,13),
  x = as.factor(c(rep('Control',8),rep('Starved',8)))
)

p <- ggplot(forPlot, aes(x=x, y=y)) + geom_boxplot()

p + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.8) +
  ylab("Cells/animal (x10E4)") + 
  xlab("Condition") +
  ggtitle("Cells per Animal")

#All pairwise fold-changes, get std dev
control <- c(23.5,25.5,25,27.5,29.5,30,33.5,28.5)
starved <- c(13,13.5,11,10.5,11.5,12.5,11.5,13)

f1 <- function(x, y) list(x/y)
out1 <- outer(control, starved, FUN = Vectorize(f1)) 

mean(as.numeric(out1))

sd(as.numeric(out1))
