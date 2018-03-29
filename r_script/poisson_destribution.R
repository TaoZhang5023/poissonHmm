

dist <- aggregate(raw_data$countAE, by=list(date=raw_data$date), FUN=sum)
library(plyr)
dist_day <- count(dist, 'x')
plot(dist_day)
