//
//  ArticleTheme.swift
//  PommeNews
//
//  Created by Jonathan Duss on 16.07.18.
//  Copyright © 2018 Swizapp. All rights reserved.
//

import Foundation
import JoLibrary

public struct ArticleTheme: Hashable, Equatable {
    
    public static let iPhone = ArticleTheme(key: "iphone")
    public static let iPad = ArticleTheme(key: "ipad")
    public static let appleWatch = ArticleTheme(key: "appleWatch")
    public static let appleTV = ArticleTheme(key: "appleTV")
    public static let mac = ArticleTheme(key: "mac")
    public static let otherAppleProduct = ArticleTheme(key:"otherAppleProduct")
    public static let pc = ArticleTheme(key: "pc")
    public static let otherPhone = ArticleTheme(key: "otherPhone")
    public static let otherTablet = ArticleTheme(key: "otherTablet")
    public static let otherWear = ArticleTheme(key: "otherWear")
    public static let androidPhone = ArticleTheme(key: "androidPhone")
    public static let androidTablet = ArticleTheme(key: "androidTablet")
    public static let androidWear = ArticleTheme(key: "androidWear")
    public static let surface = ArticleTheme(key: "surface")

    public static let android = ArticleTheme(key: "android")
    public static let ios = ArticleTheme(key: "ios")
    public static let windows = ArticleTheme(key: "windows")
    public static let macos = ArticleTheme(key: "macos")
    public static let otherOS = ArticleTheme(key: "otherOS")

    public static let apps = ArticleTheme(key: "apps")
    public static let game = ArticleTheme(key: "game")

    public static let computer = ArticleTheme(key: "computer")
    public static let smartphone = ArticleTheme(key: "smartphone")
    public static let tablet = ArticleTheme(key: "tablet")
    public static let watch = ArticleTheme(key: "watch")
    public static let otherProduct = ArticleTheme(key: "otherProduct")
    public static let smartCar = ArticleTheme(key: "smartCar")
    public static let smartHome = ArticleTheme(key: "smartHome")
    public static let speaker = ArticleTheme(key: "speaker")

    public static let apple = ArticleTheme(key: "apple")
    public static let microsoft = ArticleTheme(key: "microsoft")
    public static let google = ArticleTheme(key: "google")
    public static let samsung = ArticleTheme(key: "samsung")
    public static let amazon = ArticleTheme(key: "amazon")
    public static let spotify = ArticleTheme(key: "spotify")
    public static let netflix = ArticleTheme(key: "netflix")
    public static let facebook = ArticleTheme(key: "facebook")
    public static let amazonPrime = ArticleTheme(key: "amazonPrime")
    public static let hbo = ArticleTheme(key: "hbo")
    public static let disneyPlus = ArticleTheme(key: "disneyPlus")

    
    public static let appleMusic = ArticleTheme(key: "appleMusic")
    public static let appleTVplus = ArticleTheme(key: "appleTVplus")
    public static let appleNews = ArticleTheme(key: "appleNews")
    public static let applePay = ArticleTheme(key: "applePay")
    public static let icloud = ArticleTheme(key: "icloud")
    public static let appleServices = ArticleTheme(key: "appleServices")
    public static let services = ArticleTheme(key: "services")
    public static let paymentService = ArticleTheme(key: "paymentService")
    public static let cloudService = ArticleTheme(key: "cloudService")

    public static let audioVisualCulture = ArticleTheme(key: "audioVisualCulture")
    
    public static let audioDeviceTech = ArticleTheme(key: "musicDeviceTech")
    public static let videoDeviceTech = ArticleTheme(key: "videoDeviceTech")
    public static let photoDeviceTech = ArticleTheme(key: "photoDeviceTech")
    
    public static let videoService = ArticleTheme(key: "videoService")
    public static let audioService = ArticleTheme(key: "audioService")


    public static let promo = ArticleTheme(key: "promo")
    public static let advertisement = ArticleTheme(key: "advertisement")
    public static let rumor = ArticleTheme(key: "rumor")
    public static let security = ArticleTheme(key: "security")
    public static let privacy = ArticleTheme(key: "privacy")
    public static let patent = ArticleTheme(key: "patent")
    public static let lawsuitLegal = ArticleTheme(key: "lawsuitLegal")
    public static let technology = ArticleTheme(key: "technology")
    public static let beta = ArticleTheme(key: "beta")
    public static let economyPolitic = ArticleTheme(key: "economyPolitic")
    public static let test = ArticleTheme(key: "test")
    public static let future = ArticleTheme(key: "future")
    public static let IT = ArticleTheme(key: "IT")
    public static let AIML = ArticleTheme(key: "AIML")
    public static let networkProvider = ArticleTheme(key: "networkProvider")
    public static let study = ArticleTheme(key: "study")
    public static let crypto = ArticleTheme(key: "crypto")
    public static let health = ArticleTheme(key: "health")
    public static let cloudComputing = ArticleTheme(key: "cloudComputing")

    public static let accessory = ArticleTheme(key: "accessory")
    public static let component = ArticleTheme(key: "component")
    public static let keynote = ArticleTheme(key: "keynote")

    public static let otherSerious = ArticleTheme(key: "otherSerious")

    public static let other = ArticleTheme(key: "other")
    
    public static var allThemes: [ArticleTheme] = [
        ArticleTheme.iPhone,
        ArticleTheme.iPad,
        ArticleTheme.appleTV,
        ArticleTheme.appleWatch,
        ArticleTheme.mac,
        ArticleTheme.otherAppleProduct,
        
        ArticleTheme.pc,
        ArticleTheme.surface,
        ArticleTheme.androidPhone,
        ArticleTheme.androidTablet,
        ArticleTheme.androidWear,
        ArticleTheme.otherPhone,
        ArticleTheme.otherTablet,
        ArticleTheme.otherWear,
        
        ArticleTheme.android,
        ArticleTheme.ios,
        ArticleTheme.macos,
        ArticleTheme.windows,
        ArticleTheme.beta,
        ArticleTheme.otherOS,

        ArticleTheme.apps,
        ArticleTheme.game,

        ArticleTheme.computer,
        ArticleTheme.smartphone,
        ArticleTheme.tablet,
        ArticleTheme.watch,
        ArticleTheme.smartCar,
        ArticleTheme.smartHome,
        ArticleTheme.speaker,
        ArticleTheme.otherProduct,

        ArticleTheme.amazon,
        ArticleTheme.apple,
        ArticleTheme.facebook,
        ArticleTheme.google,
        ArticleTheme.microsoft,
        ArticleTheme.samsung,
        

        ArticleTheme.audioVisualCulture, //
        
        ArticleTheme.videoDeviceTech, // AppleTV, TV, tech for video, apps of edition
        ArticleTheme.audioDeviceTech, // Speaker, tech for audio, apps of edition
        ArticleTheme.photoDeviceTech, // Test of camera, camera tech, sensor, etc, apps of edition

        ArticleTheme.audioService, //deezer, spotify, etc streaming
        ArticleTheme.videoService, //netflix, prime, etc
        
        ArticleTheme.cloudService,
        ArticleTheme.paymentService,
        ArticleTheme.services,

        ArticleTheme.appleMusic,
        ArticleTheme.appleNews,
        ArticleTheme.appleTVplus,
        ArticleTheme.applePay,
        ArticleTheme.icloud,
        ArticleTheme.appleServices,
        
        ArticleTheme.spotify,
        ArticleTheme.netflix,
        ArticleTheme.hbo,
        ArticleTheme.amazonPrime,
        ArticleTheme.disneyPlus,
        
        
        ArticleTheme.promo,
        ArticleTheme.advertisement,
        ArticleTheme.rumor,
        ArticleTheme.keynote,
        ArticleTheme.test, // Tests or comparison
        ArticleTheme.future,
        ArticleTheme.study,

        ArticleTheme.security,
        ArticleTheme.privacy,
        ArticleTheme.technology,
        ArticleTheme.patent,
        ArticleTheme.lawsuitLegal,
        ArticleTheme.economyPolitic,
        ArticleTheme.health,
        ArticleTheme.crypto,
        ArticleTheme.AIML, // Machine Learning et cie
        ArticleTheme.IT, // Other IT news
        ArticleTheme.networkProvider,
        ArticleTheme.cloudComputing,

        ArticleTheme.component, // Any bind of component for tech (ssd, )
        ArticleTheme.accessory, // Accessory for devices (usb cable, color, etc)
        
        ArticleTheme.otherSerious,
        ArticleTheme.other]
    
    public let key: String
    
    public init(key: String) {
        self.key = key
    }
    
    public var localized: String {
        return self.key.localized
    }
    
    public func title(forLanguageCode languageCode: String) -> String {
        return Locale.current.localizedString(forLanguageCode: languageCode) ?? "" // TODO
    }
    
    public static func == (themeLeft: ArticleTheme, ThemeRight: ArticleTheme) -> Bool {
        return themeLeft.key == ThemeRight.key
    }
    
    public static func != (themeLeft: ArticleTheme, ThemeRight: ArticleTheme) -> Bool {
        return themeLeft.key != ThemeRight.key
    }
}
