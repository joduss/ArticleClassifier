//
//  ArticlesFetcher.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 28.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation
import ArticleClassifierCore
import RssClient

class ArticleFetcher {
    
    func fetchArticles(of feeds: [RssPlistFeed],
                               onProgress: @escaping (Double) -> (),
                               completion: @escaping ([TCVerifiedArticle]) -> ()) {
        
        DispatchQueue(label: "fetchArticles").async {
            var fetchedArticles: [TCVerifiedArticle] = []
                        
            for feed in feeds {
                let articles = self.fetchArticle(of: feed)
                fetchedArticles.append(contentsOf: articles)
                DispatchQueue.main.async {
                    let idx = feeds.index(where: {$0.url == feed.url}) as Int? ?? 0
                    onProgress(Double(idx + 1 / feeds.count))
                }
            }
            
            
            DispatchQueue.main.async {
                completion(fetchedArticles)
            }
        }
    }
    
    private func fetchArticle(of feed:RssPlistFeed) -> [TCVerifiedArticle] {
        
        let client = RSSClient()
        let semaphore = DispatchSemaphore(value: 0)
        var articles: [RssArticlePO] = []
        
        DispatchQueue(label: "fetch").sync {
            
            client.fetch(feed: feed, completion: { result in
                switch result {
                case .success(let fetchedArticles):
                    articles = fetchedArticles
                default: break
                }
                semaphore.signal()
            })
        }
        semaphore.wait()
        
        return articles.map({TCVerifiedArticle(title: $0.title, summary: $0.summary)})
    }
}
