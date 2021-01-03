//
//  ArticleJsonConverter.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 28.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation

public class ArticleJsonConverter {

    public init() { }

    public func convertToJson(articles: [TCArticle]) -> String? {
        let encoder = JSONEncoder()

        do {
            let json = try encoder.encode(articles)
            return String(data: json, encoding: String.Encoding.utf8)
        }
        catch {}
        return nil
    }

//    public func convertToJson(articles: [TCVerifiedArticle]) -> String? {
//        do {
//            let json = ArticleJsonConverter.convertToJson(articles: articles)
//            return String(data: json, encoding: String.Encoding.utf8)
//        }
//    }
    
    public static func convertToJson(articles: [TCVerifiedArticle]) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        return try! encoder.encode(articles.map({$0.toDto()}))
    }
    
    public static func verifiedArticleFromJson(jsonData: Data) -> [TCVerifiedArticle] {
        let decoder = JSONDecoder()
        
        return (try! decoder.decode([TCVerifiedArticleDTO].self, from: jsonData))
            .map({TCVerifiedArticle(dto: $0)})
    }
}
