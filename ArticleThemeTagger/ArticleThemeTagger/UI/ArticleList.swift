//
//  ArticleList.swift
//  ArticleThemeTagger
//
//  Created by Jonathan Duss on 26.04.20.
//  Copyright © 2020 ZaJo. All rights reserved.
//

import Foundation
import ArticleClassifierCore

class ArticleList {
    
    private let articleHashes: [Int]
    private let articleDictionary : [Int: TCVerifiedArticle]
    
    var first: TCVerifiedArticle {
        return articleDictionary[articleHashes.first!]!
    }
    
    var count: Int {
        return articleDictionary.count
    }
    
    var isEmpty: Bool {
        return articleHashes.isEmpty
    }
    
    var articles: [TCVerifiedArticle] {
        return Array(articleDictionary.values)
    }
    
    var orderedArticles: [TCVerifiedArticle] {
        var articles = [TCVerifiedArticle]()
        
        for hash in articleHashes {
            articles.append(articleDictionary[hash]!)
        }
        
        return articles
    }
    
    var randomArticle: TCVerifiedArticle {
        return articles[Int.random(in:0..<articles.count)]
    }
    
    var randomArticleWithoutTheme: TCVerifiedArticle {
        let articlesWithoutTheme = Array(articleDictionary.filter({$0.value.themes.isEmpty}))
        return articlesWithoutTheme[Int.random(in:0..<articlesWithoutTheme.count)].value
    }
    
    init() {
        articleHashes = []
        articleDictionary = [:]
    }
    
    init(articles: [TCVerifiedArticle]) {
        
        var hashes: [Int] = []
        var articlesDic: [Int: TCVerifiedArticle] = [:]
        
        for article in articles {
            let hash = article.titleSummaryHash
            
            hashes.append(hash)
            articlesDic[hash] = article
        }
        
        articleHashes = hashes
        articleDictionary = articlesDic
    }
    
    init(articleDictionary: [Int: TCVerifiedArticle], hashes: [Int]) {
        self.articleHashes = hashes
        self.articleDictionary = articleDictionary
    }
    

    
    func index(of article: TCVerifiedArticle) -> Int {
        return articleHashes.firstIndex(of: article.titleSummaryHash)!
    }
    
    
    func filteredByMissingThemes(_ themes: [String]) -> ArticleList {
        guard themes.count >= 2 else {
            return ArticleList(articleDictionary: articleDictionary, hashes: articleHashes)
        }
        
        var filteredHashes = [Int]()
        var filteredArticleDic = [Int: TCVerifiedArticle]()
        
        for hash in articleHashes {
            let article = articleDictionary[hash]!
            
            if (themes.count >= 2) {
                let intesection = article.verifiedThemes.intersection(with: themes)
                if intesection.count != themes.count, intesection.count > 0 {
                    filteredHashes.append(hash)
                    filteredArticleDic[hash] = article
                }
            }
        }
        
        return ArticleList(articleDictionary: filteredArticleDic, hashes: filteredHashes)
    }
    
    func next(after article: TCVerifiedArticle) -> TCVerifiedArticle {
        let idx = articleHashes.firstIndex(of: article.titleSummaryHash)! + 1
        
        guard idx < articles.count else {
            return article
        }
        
        return articleDictionary[articleHashes[idx]]!
    }
    
    func previous(before article: TCVerifiedArticle) -> TCVerifiedArticle {
        let idx = articleHashes.firstIndex(of: article.titleSummaryHash)! - 1
        
        guard (idx > 0) else {
            return article
        }
        
        return articleDictionary[articleHashes[idx]]!
    }
    
    func get(at idx: Int) -> TCVerifiedArticle {
        return articleDictionary[articleHashes[idx]]!
    }
}
