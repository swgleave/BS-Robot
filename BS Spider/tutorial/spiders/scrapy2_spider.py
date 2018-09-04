import scrapy


class QuotesSpider(scrapy.Spider):
    name = "simmons"

    def start_requests(self):
        urls = [
            'http://www.espn.com/espn/page2/story?page=simmons/subject/archive'
            #'http://proxy.espn.com/espn/page2/story?page=simmons/021003'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        for href in response.css('a::attr(href)'):
            url = response.urljoin(href.extract())
            yield scrapy.Request(url, callback = self.parse_contents)

    def parse_contents(self, response):
        #one = response.css('div::attr(sp-story)')
        for quote in response.css('div.sp-story):
            yield {
                "test":quote.extract()
        }
        for test in response.css('div.storybody):
            yield {
                "test":test.extract()
        }
    #there is more i am not picking up, other stuff not in any tags but in the story div