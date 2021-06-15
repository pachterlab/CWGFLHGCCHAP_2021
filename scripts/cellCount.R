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