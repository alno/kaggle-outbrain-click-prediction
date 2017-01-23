# """
# Here we create a feature keyed on display_id, where for each display, we hash the interaction
# of the display document, and each of the ads on the display.
# We use this feature by joining it back to the clicks_train and clicks_test files, so that
# each row has information on the ads it it is competing against.
# To run this script, execute the following command from the main folder :
#         Rscript final/feat_disp_ad_doc_others.R
# """


cat("Set up packages")
rm(list=ls())
if (!require("data.table")) install.packages("data.table")
library(data.table)
gc()

################################################################################################
# ad_id & document interaction per display
################################################################################################

D = 2^22  # Hash value

input_dir = Sys.getenv("INPUT", "../input")

cat("load and join the clicks and events data")
ctrnraw = fread(paste0("gunzip -c ", input_dir, "/clicks_train.csv.gz"), select=c("display_id", "ad_id"))
ctstraw = fread(paste0("gunzip -c ", input_dir, "/clicks_test.csv.gz"), select=c("display_id", "ad_id"))
craw = rbind(ctrnraw, ctstraw)
rm(ctrnraw, ctstraw); gc()
event = fread(paste0("gunzip -c ", input_dir, "/events.csv.gz"), select=c("display_id", "document_id"))
craw = merge(craw, event, all.x = T, by = "display_id")
rm(event); gc()

cat("Get the interaction of the ad and the dipslay documents")
# We add a '99' to avoid colisions; eg. if we just paste them together,
# the diplay_id '1' and ad_id '1000' would get the same value as display_id '1100' and ad_id '1'
craw[,document_ad_id:=paste0(document_id, "99", ad_id, sep="")]
craw[,`:=`(document_id=NULL, ad_id=NULL)]

cat(" We do not want to see rare events, therefore we exclude any documents see less than 100 times.")
craw[,ct:=.N, by="document_ad_id"]
craw = craw[ct>99]
craw[,ct:=NULL]
gc()

# Because of how we created the interaction above, we have some very large numbers in the interaction.
# to reduce space, lets hash these
craw[,document_ad_id:=as.numeric(document_ad_id)%%D]

cat("Now we aggregate each interaction per display")
setkeyv(craw, "display_id")
craw = craw[,(paste0(document_ad_id, collapse = " ")), by=display_id]
setnames(craw, c("display_id", "document_ad_id"))
setkeyv(craw, "display_id")

cat("Write out the file")
write.csv(craw, gzfile("cache/doc_ad_others.csv.gz"), row.names = F, quote = F)
gc()

