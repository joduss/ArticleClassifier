//
//  FeedForManualTag.swift
//  FetchNewArticleToJson
//
//  Created by Jonathan Duss on 27.08.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation

struct FeedForManualTag: Encodable {
    var title: String
    var summary: String
    var themes: [String] = []
}
