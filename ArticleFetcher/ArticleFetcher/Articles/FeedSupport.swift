//
//  FeedSupport.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 28.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation
import RssClient

class FeedSupport {
    
    private let supportedFeedPlist: String
    
    init(supportedFeedPlist: String) {
        self.supportedFeedPlist = supportedFeedPlist
    }
    
    func getFeedPO() -> [RssPlistFeed] {
        let decoder = PropertyListDecoder()
        let sitesPlistPath = Bundle.main.url(forResource: supportedFeedPlist, withExtension: "plist")!
        do {
            let sitesPlist = try Data(contentsOf: sitesPlistPath)
            let sites = try decoder.decode([RssPlistFeed].self, from: sitesPlist)
            return sites
        }
        catch {
            return []
        }
    }
    
    
    
    
}
