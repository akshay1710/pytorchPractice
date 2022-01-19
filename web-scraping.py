from autoscraper import AutoScraper

amazon_url = "https://www.amazon.in/s?k=wireless+earphone&ref=nb_sb_noss_2"

wanted_list = ["₹1,499","boAt Rockerz 255 Pro+ Bluetooth Wireless in Ear Earphones with Mic (Teal Green)", "402,308"]
#
scraper =AutoScraper()
result = scraper.build(amazon_url, wanted_list)
print(result)
#
group_result =scraper.get_result_similar(amazon_url, grouped=True)
print(group_result)

scraper.set_rule_aliases({'rule_8opm':'Title','rule_3bzo':'Price'})
scraper.keep_rules(['rule_q4do','rule_hbsb'])
final_result = scraper.save('amazon-search')
print(result['Title'])
