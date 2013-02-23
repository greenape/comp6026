require(lattice)
require(ggplot2)
fig1 <- read.csv("dump_fig_1.csv", header=TRUE)
png("fig1.png")
print(with(fig1, levelplot(Fixated ~ I(as.ordered(Size))*Time, colorkey=FALSE, col.regions=colors()[c(1, 261)], xlab=list(label="Group Size", cex=2), ylab=list(label="Time spent in groups before mixing", cex=2), scales=list(cex=2))))
dev.off()
fig2 <- read.csv("dump_fig_2.csv", header=TRUE)
large = (fig2[122:242,] +  fig2[1:121,])
large$Genotype = "Large"
selfish = (fig2[1:121,] + fig2[243:363,])
selfish$Genotype = "Selfish"
fig3_1 = rbind(large, selfish)
fig3_1$Generation = fig3_1$Generation/2

png("fig3_1.png")
print(with(fig3_1,xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()
fig2 <- subset(fig2, fig2$Generation < 120)
png("fig3.png")
print(with(fig2, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()

png("fig3_1.png")
print(with(fig3_1,xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()

fig3 <- read.csv("dump_fig_3.csv", header=TRUE)
fig3 <- subset(fig3, fig3$Generation < 120)
png("fig2.png")
print(with(fig3, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()
fig4 <- read.csv("dump_fig_4.csv", header=TRUE)
p = as.ordered(fig4$GTime/fig4$TotalTime)
png("fig4.png")
print(ggplot(fig4, aes(x=p,y=TotalTime, group=Size, fill=I(as.ordered(Size)))) + geom_bar(stat="identity",position="dodge") + scale_fill_discrete(name="Size") + theme_bw(24) + opts(panel.grid.major=theme_blank(),panel.grid.minor=theme_blank()) + xlab("Proportion of time in group") +  ylab("Total time"))
dev.off()
fig5 <- read.csv("dump_fig_5.csv", header=TRUE)
png("fig5.png")
print(with(fig5, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2),par.settings = simpleTheme(col=c("black", "grey","red","green")))))
dev.off()
fig6 <- read.csv("dump_fig_6.csv", header=TRUE)
png("fig6.png")
print(with(fig6, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()
fig7 <- read.csv("dump_fig_7.csv", header=TRUE)
png("fig7.png")
print(with(fig7, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2),par.settings = simpleTheme(col=c("red", "green","black","grey")))))
dev.off()
fig8 <- read.csv("dump_fig_8.csv", header=TRUE)
png("fig8.png")
print(with(fig8, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()
fig9 <- read.csv("dump_fig_9.csv", header=TRUE)
png("fig9.png")
print(with(fig9, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()


fig10 <- read.csv("dump_fig_10_0.csv", header=TRUE)
for(i in 1:49) {
	name <- paste(c("dump_fig_10_",i,".csv"), collapse="")
	tmp <- read.csv(name, header=TRUE)
	fig10$Proportion <- fig10$Proportion + tmp$Proportion
}
fig10$Proportion = fig10$Proportion/50

png("fig10.png")
print(with(fig10, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()


fig12 <- read.csv("dump_fig_12_0.csv", header=TRUE)
for(i in 1:49) {
	name <- paste(c("dump_fig_12_",i,".csv"), collapse="")
	tmp <- read.csv(name, header=TRUE)
	fig12$Proportion <- fig12$Proportion + tmp$Proportion
}
fig12$Proportion = fig12$Proportion/50
png("fig12.png")
print(with(fig12, xyplot(Proportion ~ Generation, groups=Genotype, type='l', auto.key=TRUE,scales=list(cex=2),xlab=list(cex=2),ylab=list(cex=2))))
dev.off()

fig11 <- read.csv("dump_fig_11.csv", header=TRUE)
png("fig11.png")
fig11$GTime <- round(fig11$GTime, 1)
print(with(fig11, levelplot(Fixated ~ I(as.ordered(GTime))*I(as.ordered(TotalTime)), colorkey=FALSE, col.regions=colors()[c(1, 261)], xlab=list(label="Proportion of time in group",cex=2), ylab=list(label="Total time", cex=2))))
dev.off()