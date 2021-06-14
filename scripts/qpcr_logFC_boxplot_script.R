library(ggplot2)

forPlot <- data.frame(
  y = c(-0.550029129,0.867773872,1.804574678,
        -5.163731173,1.361560401,1.679851352,
        3.327795857,5.93472593,5.764231042,
        3.507704903,4.434548257,4.714380453,
        4.520923637,5.823632749,5.603789425,
        5.748125605,6.930054904,6.065120301),
  x = as.factor(c(rep('SW',6),rep('DI',6),rep('KCl',6)))
)

p <- ggplot(forPlot, aes(x=x, y=y)) + geom_boxplot()

p + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.8) +
  ylab("log2FC") + 
  xlab("Condition") +
  ggtitle("XLOC_006729")
# + geom_jitter( position=position_jitter(0.2))



# XLOC_006558

forPlot2 <- data.frame(
  y = c(0.190644497,0.259482775,0.160837045,
        -0.241647826,-0.285551943,-0.083764548,
        2.142852123,2.224409468,2.202186809,
        1.486983255,1.549747265,1.714934547,
        2.127149285,2.231515727,2.20616151,
        2.691349408,2.784090815,2.805291198),
  x = as.factor(c(rep('SW',6),rep('DI',6),rep('KCl',6)))
)

p2 <- ggplot(forPlot2, aes(x=x, y=y)) + geom_boxplot()

p2 + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.8) +
  ylab("log2FC") + 
  xlab("Condition") +
  ggtitle("XLOC_006558")
# + geom_jitter( position=position_jitter(0.2))


# XLOC_000601

forPlot3 <- data.frame(
  y = c(-0.257287905, -0.08877278,-0.02193534,
        0.238797208,0.041089184,0.088109632,
        3.935969267,4.142343399,3.879276829,
        1.590727425,1.470748785,1.567825477,
        2.970733583,3.19438694,3.070661272,
        3.433067061,3.17564875,3.110480364),
  x = as.factor(c(rep('SW',6),rep('DI',6),rep('KCl',6)))
)

p3 <- ggplot(forPlot3, aes(x=x, y=y)) + geom_boxplot()

p3 + geom_dotplot(binaxis='y', stackdir='center', dotsize=0.8) +
  ylab("log2FC") + 
  xlab("Condition") +
  ggtitle("XLOC_000601")
# + geom_jitter( position=position_jitter(0.2))
