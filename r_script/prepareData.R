library(magrittr)
library(lubridate)
library(dplyr)
library(data.table)

# Set Variable
OBS_OUTPUT = "data/obs_6m.csv"
LENGTH_OUTPUT = "data/length_6m.csv"
START_DATE = "2000-01-01"
END_DATE = "2018-12-31"
DURATION = "182 days"

# Copy data from source data
raw_data <- Event

# Filter out PN05*, PN06* only and clean data
ids <- unique(raw_data[raw_data$category2 %like% "P.N0", "id"])
raw_data <- raw_data[raw_data$id %in% ids, c("id", "type", "date")]
raw_data <- raw_data[c("id", "type", "date")]

# Create a sequence of dates
start_date <- seq(as.Date(START_DATE), to=as.Date(END_DATE), by = DURATION)
date_range <- as.data.frame(start_date)
rm(start_date)

# Count events for each days
raw_data$countP <- rowSums(raw_data[-1] == "P")
raw_data$countM <- rowSums(raw_data[-1] == "M")
raw_data$countA <- rowSums(raw_data[-1] == "A")
raw_data$countE <- rowSums(raw_data[-1] == "E")
raw_data$countAE <- raw_data$countA + raw_data$countE
raw_data$countA <- NULL
raw_data$countE <- NULL
raw_data$type <- NULL
raw_data$date <- as.Date(raw_data$date)

# Group the data by DURATION
grouped_date <- cut(raw_data$date, breaks = c(date_range$start_date), include.lowest=T)
grouped_date <- as.data.frame(grouped_date)
output <- cbind(raw_data, grouped_date)
rm(grouped_date)
output$type <- NULL
output$date <- as.Date(output$grouped_date)
output$grouped_date <- NULL
output <- output %>% group_by(id, date) %>% 
  dplyr::summarize(countP=sum(countP), countM=sum(countM), countAE=sum(countAE))

# Fill in missing date range
output <- do.call(
  rbind, by(
    output,
    output$id,
    function(x) {
      out <- merge(
        data.frame(
          id=x$id[1],
          date=seq.Date(min(x$date),max(x$date),by=DURATION)
        ),
        x,
        all.x=TRUE
      )
      out$countM[is.na(out$countM)] <- 0
      out$countP[is.na(out$countP)] <- 0
      out$countAE[is.na(out$countAE)] <- 0
      out
    }  
  )
)
rownames(output) <- NULL

# Cut the first and last date range for each patient
# output <- output %>% 
#   group_by(id) %>% 
#   mutate(rank = 1:length(date)) %>% 
#   filter(1 < rank & rank < max(rank)) %>%
#   mutate(rank = NULL)

# Count events within dates sequence
length <- data.frame(table(output$id))

# Export csv files
write.csv(length, file=LENGTH_OUTPUT)
write.csv(output, file=OBS_OUTPUT)

