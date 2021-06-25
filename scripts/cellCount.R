library(ggplot2)

forPlot <- data.frame(
  y = c(47,51,50,55,59,60,67,57,26,27,22,21,23,25,23,26),
  x = as.factor(c(rep('Control',8),rep('Starved',8)))
)

p <- ggplot(forPlot, aes(x=x, y=y)) + geom_boxplot()

p + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.8) +
  ylab("Cells/ml (x10E4)") + 
  xlab("Condition") +
  ggtitle("Cells per Organism")

#All pairwise fold-changes, get std dev
control <- c(47,51,50,55,59,60,67,57)
starved <- c(26,27,22,21,23,25,23,26)

f1 <- function(x, y) list(x/y)
out1 <- outer(control, starved, FUN = Vectorize(f1)) 

mean(as.numeric(out1))

sd(as.numeric(out1))

